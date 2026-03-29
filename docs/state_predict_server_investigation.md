# state_predict 服务端排查说明

## 1. 背景

当前外部集成方观测到如下现象：

- 状态预测模型在运行时几乎总是输出 `UT`
- 但从外部链路观测看，实际已经出现了明显的队列累积和拥塞丢包
- 这与 `state_predict` 仓库中对状态模型的预期不符

本说明用于在**只有 `state_predict` 仓库代码、不了解具体业务场景**的服务器环境中开展排查。

换句话说，排查目标不是直接修业务代码，而是回答下面几个问题：

1. 当前导出的状态模型到底是什么版本、从哪个 checkpoint 来的
2. 当前 ONNX 模型的输入特征定义、维度、归一化参数是否自洽
3. 当前模型是否本来就不擅长区分 `UT/QU/EB`
4. “几乎全是 `UT`”是模型能力问题、导出物问题，还是训练/部署定义不一致问题

## 2. 当前最可能的问题

按优先级排序，当前最值得怀疑的是下面几类问题。

### 2.1 导出的 `state_prediction_no_validation.onnx` 可能来自错误的 checkpoint

这是最高优先级问题。

`experiments/no_validation` 目录中的文档表明：

- 训练是两阶段的
- `epoch_5` 仍处于阶段1，偏向 `NH vs Other`
- `best_model.pth` 可能只是“训练集 loss 最优”，并不代表 4 分类效果最好
- 文档里明确指出：
  - `epoch_10` 的 `NH recall` 最高
  - `epoch_15` 是“整体最平衡”的 checkpoint

需要重点核查：

- 当前 `state_prediction_no_validation.onnx` 是从哪个 checkpoint 导出的
- 是否直接从 `best_model.pth` 导出
- 如果是，`best_model.pth` 是否实际上对应阶段1或非最佳 4 分类模型

如果当前 ONNX 是从不合适的 checkpoint 导出的，出现“非 NH 类几乎塌缩到单一类别”的现象是可能的。

### 2.2 仓库文档和真实导出物之间可能存在版本漂移

虽然最新导出包已经附带更完整的 `metadata/provenance/scaler` 文件，但仍建议核查仓库里不同文档、脚本和导出物是否完全一致，尤其体现在：

- 是否仍有旧文档描述旧版 `18` 维输入
- 当前导出包声明的 `15` 维输入，是否和训练/导出脚本保持一致
- `README`、`metadata`、`provenance`、`src/features.py` 之间的特征顺序和 feature set 是否一致

这意味着：

- 当前 ONNX 的真实输入语义未必和文档完全一致
- 当前 scaler、特征顺序、输入维度，可能只是在“部分文件中正确”

如果训练定义、导出说明、模型元数据三者不一致，那么外部集成方即使照文档接入，也可能得到错误行为。

### 2.3 修复数据泄露后，模型对 `UT/QU/EB` 的区分能力本来就明显下降

`DATA_LEAKAGE_FIX_REPORT.md` 中明确说明：

- 旧版模型把 `min_queue/max_queue/queue_max_depth` 当作输入，属于数据泄露
- 修复后：
  - `Macro F1` 明显下降
  - `UT` 与 `QU` 的混淆显著增加
  - `EB` 表现下降最明显
  - `NH` 相对稳定

这说明：

- 当前状态模型不能被视为“可靠的 4 分类器”
- 尤其不要默认认为它能稳定地区分 `UT/QU/EB`

因此，“线上有队列和丢包，但状态模型不打出 QU/EB”不一定完全是接入错误，也可能是模型本身的真实能力边界。

### 2.4 训练数据定义和部署侧输入分布可能不一致

即使输入特征名字一样，模型也可能因为数据域漂移而失效。

例如下面这些量，在训练集和部署环境中可能具有不同统计分布或语义：

- `adjusted_rtt`
- `response_interval`
- `short_loss_rate`
- `long_loss_rate`
- `short_send_cnt`
- `long_send_cnt`
- `cwnd`
- `pkt_in_fly`

尤其需要注意：

- 训练数据来自离线 `v4` 数据集
- 这些特征是从 CSV 中直接读取并标准化的
- 部署时这些特征是运行时实时统计得到的

如果运行时统计窗口、采样频率、单位或裁剪方式和训练时不一致，模型会很容易偏向某一类。

### 2.5 当前导出的 no-validation 模型可能过度偏向 NH 或阶段性目标，而不适合整体 4 分类部署

`experiments/no_validation` 的目标更像是：

- 尽量保持 `NH recall`
- 在完全不使用验证集的情况下找可用模型

这类模型的优化目标并不等价于“在线 4 分类最稳”。

如果实际业务更关心：

- `UT/QU/EB` 的动态切换
- 拥塞和排队的稳定识别

那么 `no_validation` 导出的模型未必是最适合部署的选择。

## 3. 需要重点阅读的文件

服务器进入 `state_predict` 仓库后，建议优先阅读以下文件。

### 3.1 模型与数据定义

- `src/features.py`
- `src/data_loader.py`
- `v4_README.md`
- `onnx_export/README.md`
- `onnx_export/state_prediction_no_validation.metadata.json`
- `onnx_export/state_prediction_no_validation.provenance.json`

重点看：

- 当前真实特征维度是 `15` 还是 `18`
- 特征顺序是什么
- 派生特征如何计算
- 训练时是否做了 StandardScaler 归一化

### 3.2 数据泄露修复结论

- `DATA_LEAKAGE_FIX_REPORT.md`

重点看：

- 泄露特征是什么
- 修复后性能下降的具体幅度
- `UT/QU/EB/NH` 哪些类别最容易混淆

### 3.3 no-validation 实验结论

- `experiments/no_validation/README.md`
- `experiments/no_validation/COMPARISON_REPORT.md`
- `experiments/no_validation/EPOCH_DETAILED_COMPARISON.md`
- `experiments/no_validation/NON_NH_ANALYSIS.md`
- `experiments/no_validation/config_no_val.yaml`

重点看：

- `epoch_10`、`epoch_15`、`epoch_30`、`epoch_50` 的差异
- 哪个 checkpoint 最适合整体部署
- `best_model.pth` 是否其实不适合导出线上模型

### 3.4 ONNX 导出逻辑

- `export_onnx.py`

重点看：

- 默认导出的是哪个 checkpoint
- 导出时使用的 config 是什么
- 生成 `state_prediction_no_validation.metadata.json` 时是否和当前 feature set 保持一致

## 4. 需要排查的核心问题清单

下面的排查项都可以只在 `state_predict` 仓库中完成。

### 4.1 确认当前 ONNX 的来源

目标：

- 确认 `onnx_export/state_prediction_no_validation.onnx` 的真实来源

要查清楚：

1. 这个文件是从哪个 checkpoint 导出的
2. 导出时间对应的是哪次实验
3. 导出时使用的是 `best_model.pth`、`checkpoint_epoch_10.pth`，还是 `checkpoint_epoch_15.pth`
4. 导出使用的 config 文件是哪一个

建议输出结论：

- `state_prediction_no_validation.onnx <- checkpoint_xxx <- config_xxx`

如果这一条无法确定，就说明当前部署模型不可追溯，不适合继续直接用于分析。

### 4.2 确认当前 ONNX 的真实输入定义

目标：

- 确认 ONNX 真正期待的输入维度和特征顺序

要查清楚：

1. ONNX 的实际输入 shape 是多少
2. 当前模型是按 `15` 维训练还是 `18` 维训练
3. `state_prediction_no_validation.metadata.json` 中的 `feature_names` 是否可信
4. `src/features.py` 当前默认生成的是几维特征
5. `onnx_export/README.md` 和 `Technical_Report.md` 是否仍保留旧版描述

高风险信号：

- `input_spec.shape` 与 `feature_names` 数量不一致
- `src/features.py` 与导出文档描述不一致

### 4.3 确认 scaler 与特征定义是否匹配

目标：

- 确认归一化参数对应的正是当前 ONNX 的输入特征，而不是旧版特征集合

要查清楚：

1. `data/processed/scaler.pkl` 是用哪个特征版本拟合的
2. scaler 的维度是否等于 ONNX 输入维度
3. scaler 拟合时的特征顺序是否与当前导出模型一致
4. 是否存在旧版 `18` 维 scaler 被误用于新版 `15` 维模型的情况

建议重点验证：

- scaler 的 `mean_` 和 `scale_` 长度
- 对应的 `feature_names`

### 4.3.1 额外检查特征单位是否一致

目标：

- 确认部署侧构造特征时，`adjusted_rtt` 和 `response_interval` 等量的单位与训练/导出时一致

要查清楚：

1. `src/features.py` 中基础特征是否直接使用原始 RTT/响应间隔
2. scaler 中 `adjusted_rtt`、`response_interval` 的均值量级是否对应微秒而非毫秒
3. 外部部署方是否把这些字段额外除以 `1000` 或做了其他缩放

这是高优先级检查项，因为单位错位会让归一化后的输入整体偏离训练分布，即使特征顺序和维度都正确，模型也可能长期塌向单一类别。

### 4.4 确认 `no_validation` 模型是否本就不适合在线 4 分类

目标：

- 判断当前导出的 no-validation ONNX 是否天然偏向某些类别

要查清楚：

1. `epoch_10`、`epoch_15`、`epoch_30`、`epoch_50` 的混淆矩阵差异
2. 哪个 checkpoint 在 `UT/QU/EB` 上最稳
3. 哪个 checkpoint 只是在 `NH recall` 上最优
4. 当前导出的 no-validation ONNX 是否可能被“选错目标”

文档已经给出的强信号：

- `epoch_15` 是整体最平衡
- `epoch_10` 更偏 `NH recall`
- `best_model.pth` 未必适合线上部署

### 4.5 确认模型是否本来就难以区分 `UT/QU/EB`

目标：

- 判断线上“全是 UT”究竟是异常，还是模型能力边界的极端表现

要查清楚：

1. 修复数据泄露后，`UT/QU/EB` 的真实 F1 分别是多少
2. 哪两类最容易混淆
3. `EB` 是否本来就大量被打成 `QU`
4. `UT/QU` 是否本来就边界模糊

如果结论是：

- `UT/QU/EB` 本来就很难分

那么线上症状就不应该只从“接入错误”角度解释，而要从“模型不适合承担这个分类职责”角度重审。

### 4.6 确认训练数据标签定义与部署预期是否一致

目标：

- 判断“业务上认为有队列积累/丢包”与“训练集标签会不会打成 QU/EB”之间是否一致

需要注意：

- `v4_README.md` 中，`UT/QU` 的区分来自 `max_queue`
- 但数据泄露修复后，模型训练时已经不能再看 `max_queue`
- 这意味着训练标签的定义依赖某些运行时不可见信息
- 模型只能通过 RTT、loss、inflight、cwnd 等“间接猜测”

这会自然带来：

- 训练标签强依赖“队列真值”
- 部署特征缺少“队列真值”
- 模型只能学到不稳定代理特征

这是结构性矛盾，需要明确记录。

## 5. 建议的服务器侧实际操作

以下操作都只需要 `state_predict` 仓库。

### 5.1 记录当前模型的 provenance

建议在服务器上形成一个简短结论：

- 当前部署用的 ONNX 文件名
- 来源 checkpoint
- 导出脚本
- 对应 config
- input_dim
- feature_names
- scaler 来源

如果做不到，就先不要继续深入分析线上现象。

### 5.2 对比导出不同 checkpoint 的 ONNX

建议至少导出并对比：

1. `checkpoint_epoch_10.pth`
2. `checkpoint_epoch_15.pth`
3. 如有必要，再加 `checkpoint_epoch_30.pth`

目标不是立刻上线，而是回答：

- 哪个 checkpoint 的类别分布更合理
- 哪个 checkpoint 会更容易塌到单一类别

### 5.3 写一个最小离线推理脚本

建议在服务器中准备一个最小脚本，完成：

1. 加载当前 ONNX
2. 加载 scaler
3. 构造或读取一段 `30 x D` 的特征窗口
4. 输出 logits 和 softmax
5. 输出最终类别

这个脚本的作用是：

- 验证 ONNX 本身是否可解释
- 验证输入归一化后输出是否异常偏向 `UT`
- 作为后续和外部运行时对照的基准

### 5.4 用仓库内数据做 sanity check

如果服务器上有训练缓存或原始数据可访问，建议做：

1. 从训练集样本中抽取少量真实 `UT/QU/EB/NH`
2. 用当前 ONNX 做推理
3. 查看四类是否都能正常打出

如果在仓库自有数据上都几乎只输出 `UT`，那问题在模型/导出物本身。

如果在仓库自有数据上正常，但外部部署几乎全 `UT`，那问题更偏向：

- 输入特征分布漂移
- 部署侧特征构造/归一化不一致

## 6. 最可能的最终结论

从当前文档证据推断，最终大概率会落到下面几类结论之一。

### 结论 A：导出 checkpoint 选错

表现：

- 当前 ONNX 来自 `best_model.pth`
- 但 `best_model.pth` 并不是最适合 4 分类部署的模型

解决：

- 改用 `epoch_15` 重新导出并验证

### 结论 B：模型定义和元数据版本混乱

表现：

- `15` 维和 `18` 维描述混在一起
- `feature_names` 和真实输入不一致
- scaler 来源不明确

解决：

- 重新梳理并固定一套“训练-导出-部署”一致的配置链

### 结论 C：模型真实能力不足

表现：

- 即使用正确 checkpoint、正确 scaler，`UT/QU/EB` 区分仍不稳定

解决：

- 不再把状态模型当作强 4 分类器使用
- 重新定位：
  - `NH` 用状态模型
  - `QU` 用队列回归模型
  - `EB` 交由规则或单独建模

### 结论 D：训练/部署数据域不一致

表现：

- 仓库内部验证正常
- 外部部署时几乎全 `UT`

解决：

- 对齐外部运行时特征定义、单位、窗口、采样频率、裁剪方式
- 用真实线上窗口做离线回放验证

## 7. 建议的产出物

服务器排查结束后，建议至少形成下面三份结果。

### 7.1 模型溯源说明

内容包括：

- ONNX 文件
- checkpoint 来源
- config 来源
- scaler 来源
- 输入维度
- 特征顺序

### 7.2 checkpoint 对比表

至少比较：

- epoch_10
- epoch_15
- 当前线上模型对应 checkpoint

指标包括：

- Macro F1
- UT/QU/EB/NH F1
- NH Recall

### 7.3 最终结论

明确回答：

1. 当前“几乎全是 UT”的主要原因是什么
2. 是模型问题、导出问题，还是部署输入问题
3. 是否应该继续使用当前状态模型承担 `UT/QU/EB/NH` 四分类职责

## 8. 当前最推荐的调查顺序

如果时间有限，建议按下面顺序查：

1. 先确认 `state_prediction_no_validation.onnx` 的来源 checkpoint。
2. 再确认 ONNX 输入维度、feature list、scaler 是否一致。
3. 再看 `epoch_10` 与 `epoch_15` 的文档结论。
4. 最后判断是“导出物问题”还是“模型能力问题”。

如果只能做一件事，优先做：

- **确认当前 ONNX 是否其实是从错误 checkpoint 导出的**

这件事最有可能直接解释“为什么模型行为和预期完全不符”。
