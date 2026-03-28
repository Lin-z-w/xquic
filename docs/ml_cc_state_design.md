# ML CC State-Based拥塞控制实现计划

## 概述

将现有的 `xqc_ml_cc.c` 实现从基于队列百分比预测的模型，改为使用新的四状态(UT/QU/EB/NH)预测模型。

## 新模型信息

- **模型路径**: `third_party/state_predict/onnx_export/state_prediction_no_validation.onnx`
- **输入**: (batch_size, 30, 15) - 30个时间步 × 15个特征
- **输出**: (batch_size, 4) - 4个类别的logits分数

### 状态定义

| 状态 | 值 | 含义 | 窗口动作 |
|------|-----|------|----------|
| UT | 0 | Underutilized (带宽未充分利用) | 增长 ×1.1 |
| QU | 1 | Queue (队列堆积) | 连续3次后下降 ×0.95 |
| EB | 2 | Exceeded Bandwidth (超过带宽) | 减半 ×0.5 |
| NH | 3 | Network Handover (网络切换) | 冻结 ×0.7 |

## 关键设计

### 1. 输入特征 (15维)

| Index | Feature | 来源 |
|-------|---------|------|
| 0 | adjusted_rtt | ctl_srtt / 1000 |
| 1 | short_loss_rate | lost[0] / send[0] |
| 2 | short_lost_cnt | ctl_recent_lost_count[0] |
| 3 | short_send_cnt | ctl_recent_send_count[0] |
| 4 | long_loss_rate | (lost[0]+lost[1]) / (send[0]+send[1]) |
| 5 | long_lost_cnt | lost[0] + lost[1] |
| 6 | long_send_cnt | send[0] + send[1] |
| 7 | response_interval | (delivered_time - first_sent_time) / 1000 |
| 8 | cwnd | ml_cc->cwnd_bytes |
| 9 | pkt_in_fly | ctl_bytes_in_flight |
| 10 | rtt_diff | adjusted_rtt - prev_rtt |
| 11 | cwnd_change | cwnd / (prev_cwnd + eps), clamp [0.1, 10.0] |
| 12 | pkt_ratio | pkt_in_fly / (cwnd + eps), clamp [0, 2.0] |
| 13 | throughput_est | short_send_cnt / (response_interval/1000 + eps), clamp [0, 10000] |
| 14 | loss_rate_diff | short_loss_rate - long_loss_rate, clamp [-100, 100] |

### 2. 输出处理

1. 模型输出4个logits
2. 应用softmax得到概率
3. 如果 prob[NH] > 0.3，判定为NH
4. 否则取概率最大的状态

### 3. 状态机逻辑

```
on_ack 收到
    │
    ▼
┌─────────────────────────────────────┐
│ is_frozen == true?                  │
└─────────────────────────────────────┘
    │              │
   Yes             No
    │              │
    ▼              ▼
检查是否退出    xqc_ml_cc_run_inference()
冻结条件       获取 state_probs[4]
    │           │
    │     ┌─────┴─────┐
    │     │ softmax   │
    │     └───────────┘
    │           │
    │           ▼
    │     ┌─────────────────────┐
    │     │ prob[NH] > 0.3 ?   │
    │     └─────────────────────┘
    │        │          │
    │       Yes         No
    │        │          │
    │        ▼          ▼
    │    ┌──────┐  argmax(probs)
    │    │ NH   │      │
    │    └──┬───┘      │
    │       │    ┌─────┼────┐
    │       │    ▼     ▼    ▼
    │       │   UT    QU   EB
    │       │   │     │     │
    │       ▼   ▼     ▼     ▼
    │    冻结  ×1.1  ×0.95×连3  ×0.5
    │   cwnd=          次    eb_orig
    │   frozen   ┌─────────┘
    └────────►  记录状态，返回
```

### 4. NH冻结状态

- **进入条件**: 模型预测为NH (prob[3] > 0.3)
- **进入动作**: cwnd ×= 0.7，is_frozen = true，loss_spike_during_freeze = false
- **冻结期间**: 不调用模型，不改变cwnd
- **100ms观察窗口逻辑**:
  - 如果 elapsed < 100ms 且 short_loss_rate > 50%：标记 loss_spike_during_freeze = true，持续冻结
  - 如果 elapsed >= 100ms 且 loss_spike_during_freeze == false：退出冻结
  - 如果 loss_spike_during_freeze == true 且 short_loss_rate < 10%：退出冻结

### 5. 丢包处理

**重要**: 丢包不作为拥塞判断依据，`on_lost` 回调不执行任何窗口调整。

## 文件变更

### `src/congestion_control/xqc_ml_cc.h`

**新增宏定义:**
```c
#define XQC_ML_CC_WINDOW_SIZE             30
#define XQC_ML_CC_NUM_FEATURES            15

#define XQC_ML_CC_QU_CONSECUTIVE_THRESHOLD 3
#define XQC_ML_CC_NH_THRESHOLD            0.3f
#define XQC_ML_CC_NH_LOSS_HIGH            50.0
#define XQC_ML_CC_NH_LOSS_LOW             10.0
#define XQC_ML_CC_NH_FREEZE_CWND_FACTOR   0.7f

#define XQC_ML_CC_UT_CWND_GAIN            1.1f
#define XQC_ML_CC_QU_CWND_DECREASE        0.95f
#define XQC_ML_CC_EB_CWND_DECREASE        0.5f
```

**新增类型:**
```c
typedef enum {
    XQC_ML_CC_STATE_UT = 0,
    XQC_ML_CC_STATE_QU = 1,
    XQC_ML_CC_STATE_EB = 2,
    XQC_ML_CC_STATE_NH = 3,
    XQC_ML_CC_STATE_NONE = 4
} xqc_ml_cc_state_t;
```

**结构体新增字段:**
```c
xqc_ml_cc_state_t       last_state;
float                   state_probs[4];
int                     qu_consecutive_count;
xqc_bool_t              is_frozen;
xqc_usec_t              freeze_start_time;
float                   frozen_cwnd;
xqc_bool_t              loss_spike_during_freeze;
```

### `src/congestion_control/xqc_ml_cc.c`

**函数变更:**

| 函数 | 变更 |
|------|------|
| `xqc_ml_cc_extract_features` | 重写为15维特征，包含派生特征计算 |
| `xqc_ml_cc_normalize_features` | 适配新的scaler参数 |
| `xqc_ml_cc_softmax` | 新增，对4个logits计算softmax |
| `xqc_ml_cc_run_inference` | 处理4输出，返回softmax后的概率 |
| `xqc_ml_cc_determine_state` | 新增，基于概率判断状态 |
| `xqc_ml_cc_on_ack` | 按状态执行窗口调整 |
| `xqc_ml_cc_on_lost` | 改为空操作 |
| `xqc_ml_cc_init` | 初始化新字段 |

### `CMakeLists.txt`

**更新模型路径:**
```cmake
set(ONNX_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/state_predict/onnx_export")
set(ONNX_MODEL_PATH "${ONNX_MODEL_DIR}/state_prediction_no_validation.onnx")
```

## Scaler参数 (从scaler.pkl读取)

**Mean (15 values):**
```
89612.89, 6.80, 130.21, 1613.94, 6.75, 383.01, 4732.50, 3853.83,
1629617.78, 650.59, 1.62, 1.00, 0.000519, 1898.92, 0.052
```

**Scale (15 values):**
```
58660.21, 15.82, 445.02, 1690.24, 13.24, 993.50, 3690.58, 11710.82,
1487730.62, 511.89, 5668.53, 0.073, 0.00022, 2616.91, 9.01
```

## 可配置参数

| 参数 | 宏定义 | 默认值 | 说明 |
|------|--------|--------|------|
| QU连续阈值 | XQC_ML_CC_QU_CONSECUTIVE_THRESHOLD | 3 | 连续3次QU后开始下降窗口 |
| NH概率阈值 | XQC_ML_CC_NH_THRESHOLD | 0.3 | prob[NH] > 0.3时判定为NH |
| 冻结时长 | XQC_ML_CC_NH_FREEZE_DURATION | 100000 (100ms) | 默认冻结100ms |
| 冻结损失率高阈值 | XQC_ML_CC_NH_LOSS_HIGH | 50.0 | >50%时持续冻结 |
| 冻结损失率低阈值 | XQC_ML_CC_NH_LOSS_LOW | 10.0 | <10%时退出冻结 |
| 冻结窗口系数 | XQC_ML_CC_NH_FREEZE_CWND_FACTOR | 0.7 | 进入冻结时cwnd×0.7 |
| UT增长增益 | XQC_ML_CC_UT_CWND_GAIN | 1.1 | UT时cwnd×1.1 |
| QU下降系数 | XQC_ML_CC_QU_CWND_DECREASE | 0.95 | QU时cwnd×0.95 |
| EB下降系数 | XQC_ML_CC_EB_CWND_DECREASE | 0.5 | EB时cwnd×0.5 |
