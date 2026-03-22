# 匹配流程说明

## 总览

当前系统执行一次匹配时，分成四个阶段：

1. 选择数据集
2. 非智能体场景识别
3. 智能体动态纠偏执行
4. 结果输出与诊断

入口文件是 [smart_match.py](/d:/my_design/smart_match.py)。

## 第一阶段：选择数据集

系统首先调用 [train_improved_schema_matching.py](/d:/my_design/train_improved_schema_matching.py) 中的 `build_dataset_selection_menu()`。

这里会：

- 枚举 `SLD` 数据集
- 枚举 `SMD` 数据集
- 对 `SMD` 数据集显示训练集和匹配集两个入口

## 第二阶段：非智能体场景识别

用户选中数据集后，系统调用：

- `ImprovedSchemaMatchingTrainer.identify_dataset_scene()`

这一步不会让智能体做场景识别，而是：

1. 用 `_build_scene_payload()` 生成轻量级数据摘要
2. 调用 [scene_identifier.py](/d:/my_design/scene_identifier.py)
3. 由本地场景识别器输出：
   - `type`
   - `scene`
   - `matching_strategy`
   - `recommended_model`
   - `scene_detector`
   - `scene_confidence`
   - `scene_evidence`

### 这里采用的技术

- 轻量级数据摘要读取
- 本地 `Ollama`
- `llama3.1:latest`
- `requests` 调 `Ollama API`
- 启发式回退识别

## 第三阶段：智能体动态纠偏执行

场景识别完成后，系统调用：

- [react_schema_agent.py](/d:/my_design/react_schema_agent.py)
- `ReActSchemaAgent.run("match", ...)`

这里的智能体不是负责识别场景，而是负责：

- 根据当前匹配结果判断是否需要纠偏
- 动态调用工具
- 处理冲突、重试、清洗和报告整理

### 当前优化后的执行模式

为了提升速度，当前不是一上来就让智能体决定所有步骤，而是：

1. 先本地执行一次基础匹配
2. 先本地生成一次诊断
3. 只有当发现“需要纠偏的问题”时，才调用大模型决定下一步工具
4. 若无需纠偏，则直接整理报告输出

这样可以减少很多不必要的模型轮次。

## 智能体可调用的工具

这些工具定义在 [react_schema_agent.py](/d:/my_design/react_schema_agent.py)。

### `run_match`

作用：

- 执行一次匹配

可控参数：

- `threshold`
- `normalize_before_match`
- `top_k_per_source`

底层调用：

- `ImprovedSchemaMatchingTrainer.match_single_dataset()`

### `inspect_match_state`

作用：

- 分析当前匹配结果是否存在异常或需要纠偏

会识别的问题包括：

- `zero_candidates_after_filtering`
- `too_many_candidates`
- `field_type_incompatible`
- `one_to_many_conflict`
- `no_matches`
- `target_fields_uncovered`
- `data_quality_risk`
- `ambiguous_results`

### `lower_threshold_and_retry`

作用：

- 候选过少时降低阈值并重跑匹配

适用场景：

- 过滤后候选为零

### `raise_threshold_and_retry`

作用：

- 候选过多时提高阈值并重跑匹配

适用场景：

- 阈值过松
- 结果过多

### `clean_and_retry`

作用：

- 对脏数据进行规范化处理后再重跑匹配

适用场景：

- `Dirty` 类数据集
- 数据质量差导致结果不稳定

底层技术：

- `DataPreprocessor.normalize_data()`

### `resolve_one_to_many_conflicts`

作用：

- 对同一个源字段或源记录，如果匹配到了多个目标，只保留最高置信度的一条

适用场景：

- 一对多冲突

### `finalize_report`

作用：

- 汇总最终诊断
- 标记模糊结果需要人工确认
- 把诊断信息写入结果结构

## 第四阶段：真正的匹配算法

真实的匹配计算在 [train_improved_schema_matching.py](/d:/my_design/train_improved_schema_matching.py)。

### SLD 分支

调用：

- `predict_sld_matches()`

使用的技术：

- `RandomForestClassifier`
- `GradientBoostingClassifier`
- 多种相似度特征：
  - `SequenceMatcher`
  - `Jaccard`
  - `Levenshtein`
  - `token overlap`
  - `contextual similarity`
  - `phonetic similarity`
  - `data type compatibility`

### SMD 分支

调用：

- `predict_smd_matches()`

使用的技术：

- 字段级特征工程
- `StandardScaler`
- `GradientBoostingClassifier`
- `RandomForestClassifier`
- 候选裁剪
- `top-k` 选择

## 动态处理的八类情况

系统当前支持这类情况的自动响应：

1. 过滤后候选对为零
   - 动作：降低阈值后重试

2. 候选对过多
   - 动作：提高阈值后重过滤

3. 字段类型不兼容
   - 动作：保留结果并输出警告

4. 一对多冲突
   - 动作：自动消歧

5. 完全无匹配
   - 动作：在诊断报告中说明原因

6. 目标字段未被覆盖
   - 动作：报告未覆盖字段

7. 数据质量差
   - 动作：规范化清洗后重试

8. 结果模糊
   - 动作：标记 `review_required`

## 输出结果

最终结果会被写入：

- `matching_results_<dataset>.json`

控制台会打印：

- 场景识别结果
- 智能体动作日志
- 问题标签
- 诊断摘要
- Accuracy / Precision / Recall / F1
- 执行耗时 / 总耗时

## 当前优化点

为了减少智能体调用工具的延迟，当前系统已经做了这些优化：

- 不让智能体做场景识别
- 先本地跑基础匹配和基础诊断
- 只有真的出现可纠偏问题时才调用大模型
- 限制工具调用轮数
- 直接把工具调用日志打印到控制台

## 相关文件

- 入口：[smart_match.py](/d:/my_design/smart_match.py)
- 智能体：[react_schema_agent.py](/d:/my_design/react_schema_agent.py)
- 场景识别：[scene_identifier.py](/d:/my_design/scene_identifier.py)
- 核心训练/匹配：[train_improved_schema_matching.py](/d:/my_design/train_improved_schema_matching.py)
