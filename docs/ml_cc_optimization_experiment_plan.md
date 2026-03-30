# ML CC 性能优化实验计划

## 1. 问题现状

### 1.1 性能对比（BBR vs ML_CC）

| 指标 | BBR | ML_CC | 差距 |
|------|-----|-------|------|
| **最终 cwnd** | 4,243,417 bytes | 77,254 bytes | **55x** |
| **最终 RTT** | ~47ms | ~75ms | **1.6x** |
| **吞吐** | 高 | 低 | **显著落后** |

### 1.2 关键问题现象

#### 问题1：EB状态过度降窗
```
samples:75  cwnd:294,604 → EB判定 → 147,302 (×0.5)
samples:76  cwnd:147,302 → EB持续 → 73,651  (×0.5)
samples:77  cwnd:73,651  → EB持续 → 36,825  (×0.5)
samples:78  cwnd:36,825  → EB持续 → 18,412  (×0.5)
samples:79  cwnd:18,412  → EB持续 → 9,206   (×0.5)
```
**5次连续减半，窗口仅剩3%！**

#### 问题2：RTT异常飙升
- 正常网络RTT：~70ms
- ML_CC测量RTT：峰值达**250ms+**
- 原因：cwnd压缩到5.7K后，触发重传风暴，RTT包含RTO等待时间

#### 问题3：管道饥饿
- cwnd = 5,744 bytes（约5个MSS）
- 但 pkt_in_fly = 231（大量重传包）
- short_loss_rate = 46.15%（极高丢包率）

## 2. 问题根因分析

### 2.1 根因1：EB降窗过于激进

**当前实现：**
```c
if (ml_cc->last_state == XQC_ML_CC_STATE_EB) {
    ml_cc->cwnd_bytes *= 0.5f;  // 直接减半
}
```

**问题：**
- 无连续降窗次数限制
- 降窗幅度过大（50%）
- 缺乏快速恢复机制

### 2.2 根因2：缺乏最小窗口保护

**当前实现：**
```c
xqc_ml_cc_clamp_cwnd(ml_cc);  // 只检查上下限
```

**问题：**
- 最小窗口 = 4 * MSS (~4.8KB)
- 无针对网络状况的动态保护
- 极端情况下管道完全饥饿

### 2.3 根因3：状态切换后恢复缓慢

**当前实现：**
- EB → UT/QU切换后，依赖UT自然增长
- UT增长速率：acked * 1.1（每个ACK增长10%）
- 从5.7K恢复到300K需要大量ACK

## 3. 实验方案

### 3.1 实验目标

1. **提升吞吐**：最终cwnd达到BBR的50%以上
2. **降低时延**：平均RTT控制在80ms以内
3. **减少抖动**：RTT标准差 < 30ms
4. **保持公平性**：与BBR共存时不超过其窗口的2倍

### 3.2 实验设计

#### 实验组1：限制连续EB次数（方案A）

**改动内容：**
```c
#define XQC_ML_CC_MAX_CONSECUTIVE_EB 2

// 在EB处理逻辑中添加计数保护
if (ml_cc->eb_consecutive_count >= XQC_ML_CC_MAX_CONSECUTIVE_EB) {
    // 达到上限，不再降窗
    action = "eb_hold";
} else {
    ml_cc->cwnd_bytes *= XQC_ML_CC_EB_CWND_DECREASE;
    ml_cc->eb_consecutive_count++;
}
```

**预期效果：**
- 最多连续2次降窗，cwnd最低降至25%
- 避免极端饥饿状态
- **预期cwnd提升：10-20x**

**验证指标：**
- 记录每次EB状态的cwnd变化
- 统计连续EB次数分布
- 对比吞吐和RTT

#### 实验组2：降低降窗幅度（方案A增强）

**改动内容：**
```c
#define XQC_ML_CC_EB_CWND_DECREASE 0.75f  // 从0.5改为0.75
```

**预期效果：**
- 每次降窗只减少25%而非50%
- 更温和的拥塞响应
- **预期cwnd提升：2-4x**

**验证指标：**
- 对比降窗前后的cwnd曲线
- 观察丢包率变化（是否恶化）

#### 实验组3：设置最小cwnd阈值（方案B）

**改动内容：**
```c
#define XQC_ML_CC_MIN_CWND_BYTES (20 * XQC_ML_CC_MSS)  // ~24KB

// 在窗口调整后强制保护
if (ml_cc->cwnd_bytes < XQC_ML_CC_MIN_CWND_BYTES) {
    ml_cc->cwnd_bytes = XQC_ML_CC_MIN_CWND_BYTES;
}
```

**预期效果：**
- 保证最少20个MSS的管道利用率
- 防止完全饥饿
- 为重传保留空间
- **预期RTT降低：30-50%**

**验证指标：**
- 监控最低cwnd是否被触发
- 统计重传率和RTO次数
- RTT分布对比

#### 实验组4：EB后快速恢复（方案C）

**改动内容：**
```c
// 状态切换时检测
if (prev_state == XQC_ML_CC_STATE_EB && 
    ml_cc->last_state != XQC_ML_CC_STATE_EB) {
    // 退出EB时，恢复到初始cwnd的2倍
    ml_cc->cwnd_bytes = xqc_max(ml_cc->cwnd_bytes, 
                                ml_cc->init_cwnd_bytes * 2);
    ml_cc->eb_consecutive_count = 0;  // 重置计数器
}
```

**预期效果：**
- 缩短恢复时间
- 避免长时间停留在低cwnd
- **预期恢复时间：缩短80%**

**验证指标：**
- 记录EB→UT/QU切换后的cwnd恢复曲线
- 统计从低cwnd恢复到正常的时间

### 3.3 对照组

- **Baseline**: 当前ML_CC实现
- **BBR**: BBR算法（性能上限参考）

## 4. 实验步骤

### 4.1 环境准备

1. **测试环境**
   - 网络拓扑：客户端 ←→ 路由器 ←→ 服务器
   - 链路带宽：100Mbps
   - 基础RTT：50ms
   - 丢包率：0-5%（可变）

2. **测试工具**
   - 使用xquic test_client/test_server
   - 传输文件大小：100MB
   - 测试时长：5分钟/组

3. **数据采集**
   ```bash
   # 日志级别设置
   export XQC_LOG_LEVEL=XQC_LOG_REPORT
   
   # 运行测试
   ./test_server -l d -e &
   sleep 1
   ./test_client -s 127.0.0.1 -p 8443 -f 100MB.file
   ```

### 4.2 实验流程

```
第1轮：Baseline测试（当前ML_CC）
  ├─ 每组3次重复，取平均值
  ├─ 记录cwnd、RTT、吞吐、丢包率
  └─ 作为对照基准

第2轮：实验组1（限制连续EB次数）
  ├─ 修改代码，编译
  ├─ 相同测试条件
  └─ 对比Baseline

第3轮：实验组2（降低降窗幅度）
  └─ 同上

第4轮：实验组3（最小cwnd阈值）
  └─ 同上

第5轮：实验组4（EB后快速恢复）
  └─ 同上

第6轮：组合方案（1+2+3）
  └─ 综合最优参数

第7轮：对比BBR
  └─ 相同测试条件运行BBR
```

### 4.3 数据分析

**关键指标：**
1. **吞吐（Throughput）**
   - 计算：传输字节数 / 传输时间
   - 目标：达到BBR的50%以上

2. **平均RTT**
   - 计算：所有ACK的adjusted_rtt平均值
   - 目标：< 80ms

3. **RTT标准差**
   - 计算：RTT样本的标准差
   - 目标：< 30ms

4. **cwnd利用率**
   - 计算：实际cwnd / 理论最大cwnd
   - 目标：> 30%

5. **重传率**
   - 计算：重传包数 / 总发送包数
   - 目标：< 10%

**可视化：**
- cwnd时间序列图
- RTT分布直方图
- 吞吐对比柱状图

## 5. 预期结果与风险评估

### 5.1 预期效果

| 方案 | 预期吞吐提升 | 预期RTT降低 | 风险 |
|------|-------------|-------------|------|
| 限制连续EB | 10-20x | 30-40% | 拥塞判断滞后 |
| 降低降窗幅度 | 2-4x | 20-30% | 队列堆积加剧 |
| 最小cwnd保护 | 5-10x | 30-50% | 低带宽环境不适用 |
| 快速恢复 | 恢复时间-80% | 波动减少 | 误判时过度激进 |
| **组合方案** | **15-25x** | **40-60%** | 需调参平衡 |

### 5.2 风险评估

**风险1：拥塞判断滞后**
- 缓解：保留EB状态的置信度阈值检查
- 监控：队列深度是否持续增长

**风险2：与BBR不公平竞争**
- 缓解：添加最大cwnd限制（如1MB）
- 监控：共存时的带宽分配

**风险3：低带宽环境恶化**
- 缓解：最小cwnd阈值可配置
- 监控：不同带宽下的表现

## 6. 实施计划

### 6.1 时间线

| 阶段 | 时间 | 任务 |
|------|------|------|
| 准备 | Day 1 | 环境搭建、基线测试 |
| 实验1 | Day 2 | 限制连续EB次数 |
| 实验2 | Day 3 | 降低降窗幅度 |
| 实验3 | Day 4 | 最小cwnd阈值 |
| 实验4 | Day 5 | EB后快速恢复 |
| 组合 | Day 6 | 综合方案测试 |
| 对比 | Day 7 | BBR对比测试 |
| 分析 | Day 8 | 数据分析、文档撰写 |

### 6.2 代码修改清单

```
src/congestion_control/xqc_ml_cc.h
  ├─ 新增宏定义：
  │   - XQC_ML_CC_MAX_CONSECUTIVE_EB
  │   - XQC_ML_CC_MIN_CWND_BYTES
  │   - XQC_ML_CC_EB_CWND_DECREASE (修改)
  └─ 结构体新增字段：
      - eb_consecutive_count

src/congestion_control/xqc_ml_cc.c
  ├─ xqc_ml_cc_init: 初始化eb_consecutive_count
  ├─ xqc_ml_cc_feed_features: 
  │   - 修改EB处理逻辑（添加计数保护）
  │   - 添加最小cwnd保护
  │   - 添加状态切换恢复逻辑
  └─ xqc_ml_cc_apply_qu_fine_grained_control: 可选优化
```

## 7. 成功标准

**最低成功标准：**
- 吞吐达到BBR的30%（当前<5%）
- 平均RTT < 100ms（当前~150ms峰值）

**目标成功标准：**
- 吞吐达到BBR的50%
- 平均RTT < 80ms
- RTT标准差 < 30ms

**卓越标准：**
- 吞吐达到BBR的70%
- 平均RTT < 70ms
- 尾时延(P99) < 100ms

## 8. 附录

### 8.1 相关代码位置

- 当前ML_CC实现：`src/congestion_control/xqc_ml_cc.c`
- 日志输出：`xqc_ml_cc_feed_features`函数
- BBR对比：`src/congestion_control/xqc_bbr.c`

### 8.2 关键日志字段

```
|ml_cc|state_update|prev_state:UT|state:EB|prob_ut:0.299|prob_qu:0.272|prob_eb:0.426|cwnd:294604
|ml_cc|cwnd_adjust|action:eb_decrease|cwnd_before:294604|cwnd_after:147302
```

### 8.3 参考文献

1. BBR: Congestion-Based Congestion Control, ACM Queue 2016
2. ML_CC Design Doc: `docs/ml_cc_state_design.md`
3. QUIC Recovery RFC 9002

---

**创建日期**: 2026-03-30  
**负责人**: [待填写]  
**审核状态**: Draft
