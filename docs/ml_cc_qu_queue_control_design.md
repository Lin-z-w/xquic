# ML_CC QU 队列深度细粒度控制实现方案

## 1. 目标

当前 `ML_CC` 的状态机已经具备以下行为：

- `UT`：窗口增长
- `QU`：连续多次命中后轻微降窗
- `EB`：窗口减半
- `NH`：窗口降到 0.7 倍并冻结，100ms 后按丢包率退出

本次改造的目标不是重写整套状态机，而是只对 `QU` 状态做细粒度增强：

1. 保留现有 `UT/EB/NH` 行为不变。
2. 在 `QU` 状态下接入 `qu_queue_depth` 回归模型，预测当前队列深度百分比。
3. 引入可配置阈值 `k`，把它作为吞吐和时延的 trade-off 旋钮。
4. 当预测队列深度 `< k` 时继续增长窗口；当预测队列深度 `> k` 时开始下降窗口。
5. 增长和下降都采用渐进曲线，避免现有 “连续 N 次 QU 后再一次性降窗” 的突变行为。

## 2. 现状与约束

### 2.1 现有 `ML_CC` 实现

当前实现位于：

- [docs/ml_cc_state_design.md](D:/Workspace/xquic/docs/ml_cc_state_design.md)
- [src/congestion_control/xqc_ml_cc.c](D:/Workspace/xquic/src/congestion_control/xqc_ml_cc.c)
- [src/congestion_control/xqc_ml_cc.h](D:/Workspace/xquic/src/congestion_control/xqc_ml_cc.h)

现有状态分类模型已经使用长度为 30 的滑动窗口，输入 15 维特征，输出 4 类状态概率。

### 2.2 新队列回归模型

队列预测模型位于：

- [third_party/state_predict/onnx_export/qu_queue_depth/README.md](D:/Workspace/xquic/third_party/state_predict/onnx_export/qu_queue_depth/README.md)
- [third_party/state_predict/onnx_export/qu_queue_depth/model_metadata.json](D:/Workspace/xquic/third_party/state_predict/onnx_export/qu_queue_depth/model_metadata.json)
- [third_party/state_predict/onnx_export/qu_queue_depth/qu_queue_depth_regressor.onnx](D:/Workspace/xquic/third_party/state_predict/onnx_export/qu_queue_depth/qu_queue_depth_regressor.onnx)

根据模型文档：

- 输入形状：`(batch, 30, 15)`
- 输出形状：`(batch,)`
- 输出含义：`queue_depth_percentage`，范围 `[0, 1]`
- 只对 `QU` 状态样本有意义
- 使用独立的 `scaler.pkl`

队列深度目标定义为：

```text
queue_depth_percentage = ((max_queue + min_queue) / 2) / queue_max_depth
```

### 2.3 一个必须处理的实现问题

当前 `xqc_ml_cc.c` 中虽然保留了 `xqc_ml_cc_normalize_features()`，但在真正喂给状态模型推理时，并没有对输入窗口做归一化后再送入 ONNX。队列回归模型同样要求输入必须经过 scaler 归一化，因此这次改造建议顺手把“双模型各自归一化”的流程补完整，否则回归输出会明显失真。

## 3. 总体设计

采用“两阶段推理 + 单状态细化控制”的方式：

```text
ACK 到达
  -> 更新 15 维特征窗口
  -> 状态分类模型推理
  -> 得到 state in {UT, QU, EB, NH}
  -> 若 state != QU:
       维持现有 UT / EB / NH 行为
  -> 若 state == QU:
       调用 qu_queue_depth 回归模型
       得到 queue_depth_pct in [0, 1]
       基于阈值 k 计算渐进增减幅度
       执行平滑的 ACK 驱动窗口调节
```

核心思路是：

- `UT/EB/NH` 继续由状态分类模型主导。
- `QU` 从“单一状态”进一步拆成“队列还不深的 QU”和“队列已经偏深的 QU”。
- `k` 越小，算法越偏向低时延；`k` 越大，算法越偏向吞吐。

## 4. QU 细粒度控制策略

### 4.1 关键变量

定义：

- `q_raw`：回归模型输出的原始队列深度百分比，范围 `[0, 1]`
- `q_hat`：平滑后的队列深度估计
- `k`：目标队列阈值，范围 `(0, 1)`
- `deadzone`：阈值附近的静区，避免在 `k` 附近反复抖动

建议默认值：

- `k = 0.40`
- `deadzone = 0.05`
- `ema_alpha = 0.25`

其中：

- `k = 0.25` 更偏低时延
- `k = 0.40` 更偏平衡
- `k = 0.60` 更偏吞吐

### 4.2 队列估计平滑

由于模型 MAE 约为 `13.69%`，直接使用单次推理输出容易导致 ACK 级别抖动，建议加入 EMA：

```text
q_hat = (1 - ema_alpha) * q_hat_prev + ema_alpha * q_raw
```

并对输出做裁剪：

```text
q_raw = clamp(q_raw, 0.0, 1.0)
q_hat = clamp(q_hat, 0.0, 1.0)
```

### 4.3 控制分区

将 `QU` 状态分成三个区域：

1. `q_hat < k - deadzone`
   含义：虽然被分类为 `QU`，但队列仍低于目标阈值，可以继续增长。

2. `k - deadzone <= q_hat <= k + deadzone`
   含义：处于目标附近，尽量稳住窗口，避免不必要振荡。

3. `q_hat > k + deadzone`
   含义：队列明显高于目标，应开始下降窗口。

### 4.4 渐进曲线

为了让增长和下降都随“偏离阈值的程度”逐步增强，建议使用归一化距离加幂函数：

#### 增长侧

当 `q_hat < k - deadzone` 时：

```text
dist_up = (k - deadzone - q_hat) / max(k - deadzone, eps)
dist_up = clamp(dist_up, 0.0, 1.0)
grow_strength = pow(dist_up, gamma_up)
```

建议：

- `gamma_up = 1.5`

直观上：

- 当队列远低于 `k`，`grow_strength` 接近 1，增长更积极
- 当队列接近 `k`，`grow_strength` 逐渐接近 0，增长自动放缓

#### 下降侧

当 `q_hat > k + deadzone` 时：

```text
dist_down = (q_hat - (k + deadzone)) / max(1.0 - (k + deadzone), eps)
dist_down = clamp(dist_down, 0.0, 1.0)
drain_strength = pow(dist_down, gamma_down)
```

建议：

- `gamma_down = 1.2`

直观上：

- 当队列只略高于 `k`，下降很轻
- 当队列逼近满队列，下降幅度逐步增强

### 4.5 ACK 驱动窗口调节公式

为了贴合当前 `xquic` 的 ACK 驱动拥塞控制回调，建议使用“按 ACK 字节数缩放”的平滑加减，而不是每次都直接乘一个固定比例。

设：

- `acked = po ? po->po_used_size : XQC_ML_CC_MSS`

则：

#### 低于阈值时增长

```text
cwnd += acked * qu_growth_base * grow_strength
```

建议：

- `qu_growth_base = 1.10`

这会让 `QU` 内部的增长在“队列很浅”时接近当前 `UT` 风格，在接近 `k` 时自动变缓。

#### 高于阈值时下降

```text
cwnd -= acked * qu_drain_base * drain_strength
```

建议：

- `qu_drain_base = 1.30`

选择比增长略大的基准系数，是为了在 `QU` 状态下更快地把队列拉回 `k` 附近。

#### 静区内保持

```text
cwnd += acked * qu_hold_gain
```

建议默认：

- `qu_hold_gain = 0.05`

也可以直接取 `0`，完全保持不变。更推荐保留一个很小的正值，这样窗口不会因为长期停在静区而过于僵硬。

### 4.6 为什么不用原来的 “连续 N 次 QU 后再乘 0.95”

原始方案的问题是：

- 对 `QU` 的处理只有“有队列”和“没有动作”两档
- 无法表达队列深度与目标值之间的连续偏差
- 对 `k` 无法形成平滑的 trade-off
- 一旦触发下降，动作是突变的

回归模型接入后，`QU` 应该从“离散处罚”改为“连续调节”。因此建议把 `qu_consecutive_count` 从主控制逻辑中移除，或者仅保留为 fallback 路径。

## 5. 和现有状态机的协同

### 5.1 UT

保持现有逻辑：

```text
cwnd += acked * XQC_ML_CC_UT_CWND_GAIN
```

### 5.2 EB

保持现有逻辑：

```text
cwnd *= XQC_ML_CC_EB_CWND_DECREASE
```

### 5.3 NH

保持现有冻结逻辑不变：

- `prob[NH] > threshold` 时进入冻结
- `cwnd *= 0.7`
- 100ms 内不做模型驱动调节
- 根据 `short_loss_rate` 决定是否延长冻结

### 5.4 QU

只在 `last_state == QU` 时调用回归模型。

因此整体优先级为：

1. `NH` 冻结逻辑优先级最高
2. 非冻结时先跑状态分类
3. 只有命中 `QU` 才跑回归模型并执行细粒度控制
4. 其余状态继续走原逻辑

## 6. 代码改造建议

## 6.1 `xqc_ml_cc.h`

建议新增宏：

```c
#define XQC_ML_CC_QU_QUEUE_K                 0.40f
#define XQC_ML_CC_QU_QUEUE_DEADZONE          0.05f
#define XQC_ML_CC_QU_QUEUE_EMA_ALPHA         0.25f

#define XQC_ML_CC_QU_GROWTH_BASE             1.10f
#define XQC_ML_CC_QU_DRAIN_BASE              1.30f
#define XQC_ML_CC_QU_HOLD_GAIN               0.05f

#define XQC_ML_CC_QU_GAMMA_UP                1.5f
#define XQC_ML_CC_QU_GAMMA_DOWN              1.2f

#define XQC_ML_CC_QU_CONFIDENCE_THRESHOLD    0.50f
```

建议在 `xqc_ml_cc_t` 中新增字段：

```c
void                   *queue_onnx_session;
void                   *queue_onnx_api;

float                   queue_depth_raw;
float                   queue_depth_ema;
float                   queue_threshold_k;

int                     queue_model_ready;
int                     queue_model_enabled;
```

说明：

- 不建议复用 `onnx_session`，因为状态分类和队列回归是两个独立模型。
- `queue_depth_ema` 用于 ACK 级别平滑。
- `queue_threshold_k` 建议先作为宏默认值初始化，后续可扩展为 `cc_params` 配置项。

## 6.2 `xqc_ml_cc.c`

建议新增函数：

### 模型与特征处理

- `xqc_ml_cc_normalize_features_for_state()`
- `xqc_ml_cc_normalize_features_for_queue()`
- `xqc_ml_cc_run_state_inference()`
- `xqc_ml_cc_run_qu_queue_inference()`

原因是两个模型虽然输入维度相同，但 scaler 需要独立维护，命名上分开更不容易出错。

### QU 控制辅助函数

- `xqc_ml_cc_update_qu_queue_depth()`
- `xqc_ml_cc_calc_qu_growth_strength()`
- `xqc_ml_cc_calc_qu_drain_strength()`
- `xqc_ml_cc_apply_qu_fine_grained_control()`

建议把 `xqc_ml_cc_on_ack()` 改成如下结构：

```text
if (is_frozen)
    处理 NH 冻结退出逻辑

state = last_state

switch (state):
  UT:
    走现有 UT 增长

  QU:
    if queue model ready and sample_count >= 30:
        跑队列回归
        更新 queue_depth_ema
        基于 k 和渐进曲线做细粒度调节
    else:
        回退到旧的 QU 逻辑

  EB:
    走现有 EB 降窗

  NH:
    走现有冻结入口逻辑
```

### 回退策略

任一情况下都要能回退到旧 QU 逻辑：

- 队列模型 session 创建失败
- ONNX 推理失败
- 输出为 NaN / Inf
- 样本数不足 30
- `prob_qu` 太低

建议回退条件：

```text
if state == QU and state_probs[QU] < 0.50:
    使用旧 QU 逻辑
```

这样可以避免状态分类本身不稳定时，队列模型被误触发。

## 6.3 CMake

当前只定义了主状态模型路径：

```cmake
set(ONNX_MODEL_PATH "${ONNX_MODEL_DIR}/state_prediction_no_validation.onnx")
```

建议新增：

```cmake
set(ONNX_QUEUE_MODEL_PATH
    "${ONNX_MODEL_DIR}/qu_queue_depth/qu_queue_depth_regressor.onnx")
```

并注入宏：

```cmake
add_definitions(-DXQC_ONNX_QUEUE_MODEL_PATH="${ONNX_QUEUE_MODEL_PATH}")
```

如果后续要把 scaler 参数也编进 C 代码，建议同时增加一个离线脚本，把 `scaler.pkl` 转为 `const float mean[] / scale[]` 的头文件，避免运行时解析 pickle。

## 7. scaler 处理建议

两个模型都要求归一化，但运行时不适合解析 Python 的 `pkl` 文件，因此建议：

1. 保持现有主状态模型的 `const float scaler_mean[] / scaler_scale[]` 做法。
2. 为 `qu_queue_depth` 额外生成一套：

```c
extern const float xqc_ml_cc_qu_scaler_mean[XQC_ML_CC_NUM_FEATURES];
extern const float xqc_ml_cc_qu_scaler_scale[XQC_ML_CC_NUM_FEATURES];
```

3. 在推理前把 `history_window` 拷贝到临时 buffer 后按对应 scaler 归一化。

注意：

- 当前文档显示回归模型使用 15 维特征。
- 最新状态分类部署包已经改为同名 `metadata/provenance/scaler` 配套文件，优先以
  `state_prediction_no_validation.metadata.json` 和
  `state_prediction_no_validation.provenance.json` 为准。

因此实现时建议以两个 ONNX 模型实际导出的输入维度和当前 `xqc_ml_cc.c` 已落地的 15 维特征为准，不要直接依赖顶层 README 的描述。

## 8. 参数建议

首版建议参数如下：

| 参数 | 建议值 | 含义 |
|------|--------|------|
| `k` | `0.40` | 平衡模式 |
| `deadzone` | `0.05` | 阈值附近静区 |
| `ema_alpha` | `0.25` | 队列估计平滑系数 |
| `qu_growth_base` | `1.10` | `q < k` 时最大增长基线 |
| `qu_drain_base` | `1.30` | `q > k` 时最大下降基线 |
| `qu_hold_gain` | `0.05` | 静区内极小增长 |
| `gamma_up` | `1.5` | 增长曲线陡峭度 |
| `gamma_down` | `1.2` | 下降曲线陡峭度 |
| `prob_qu_floor` | `0.50` | 队列模型启用置信度下限 |

可以预设三档模式：

| 模式 | `k` | 预期效果 |
|------|-----|----------|
| LowLatency | `0.25` | 更快抑制排队，牺牲部分吞吐 |
| Balanced | `0.40` | 吞吐和时延折中 |
| HighThroughput | `0.60` | 允许更深队列，提升吞吐 |

## 9. 建议的日志与观测字段

为了后续调参，建议新增日志字段：

- `state`
- `prob_qu`
- `queue_raw`
- `queue_ema`
- `queue_k`
- `grow_strength`
- `drain_strength`
- `acked_bytes`
- `cwnd_before`
- `cwnd_after`
- `qu_mode`：`grow / hold / drain / fallback`

这样可以直接从日志中判断：

- 队列模型是否稳定
- `k` 是否过高或过低
- 是否出现频繁越过阈值的抖动

## 10. 验证方案

建议至少做三类验证：

### 10.1 功能验证

- `sample_count < 30` 时不触发队列模型
- 非 `QU` 状态时不触发队列模型
- `NH` 冻结期间完全跳过队列模型
- 推理失败时正确回退到旧 QU 逻辑

### 10.2 性能验证

固定相同 trace，比对：

- 吞吐量
- 平均 RTT
- P95 RTT
- P99 RTT
- 丢包率
- `cwnd` 抖动幅度

### 10.3 trade-off 验证

至少测试：

- `k = 0.25`
- `k = 0.40`
- `k = 0.60`

预期结果：

- `k` 越小，队列更浅、尾时延更低、吞吐略降
- `k` 越大，吞吐更高、排队更深、尾时延更高

## 11. 推荐落地顺序

建议按以下顺序实现，风险最小：

1. 先补齐“双模型各自归一化”的基础设施。
2. 再把队列回归模型 session 接入 `xqc_ml_cc_init()`。
3. 在 `QU` 状态下打印 `queue_raw` 和 `queue_ema`，先只观测不控窗。
4. 确认输出稳定后，再接入 `k` 控制逻辑。
5. 最后再开放 `k` 为可配置参数。

## 12. 最终建议

建议把这次改造理解为：

- 外层仍然是“四状态状态机”
- 内层只对 `QU` 做“基于队列深度的连续控制”

这样可以最大限度复用现有 `ML_CC` 框架，同时把 `k` 变成一个真正可调的吞吐/时延旋钮，而不是仅靠 `QU` 的离散惩罚去间接控制排队。
