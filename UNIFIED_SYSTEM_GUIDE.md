# 统一智能匹配系统使用说明

## 系统现状

当前系统已经统一到两条入口：

```bash
python smart_train.py
python smart_match.py
```

系统行为如下：

- 先展示可选数据集
- 由本机 `llama3.1` 结合 `LangChain + ReAct` 自动分析数据特征
- 自动选择更合适的流程来处理 `SLD` 或 `SMD`
- 调用对应训练工具或匹配工具完成任务

## 大模型与框架接入位置

- 工作流智能体：[react_schema_agent.py](/d:/my_design/react_schema_agent.py)
  - 使用 `langchain_ollama.ChatOllama`
  - 使用 `langgraph.prebuilt.create_react_agent`
  - 通过工具自动调用 `inspect_dataset`、`identify_scene`、`train_dataset`、`match_dataset`

- 场景识别器：[scene_identifier.py](/d:/my_design/scene_identifier.py)
  - 优先走 `LangChain/LangGraph` 的 ReAct 识别链路
  - 回退时才直接调用 `Ollama API`

- 训练入口：[smart_train.py](/d:/my_design/smart_train.py)
- 匹配入口：[smart_match.py](/d:/my_design/smart_match.py)

## 菜单说明

菜单中不会提前标注 “SMD/SLD 场景”。
当前只显示中性的数据集名称和简短提示。

其中 `mimic_2_omop` 会拆成两个可选项：

- `mimic_2_omop  训练集 (108 mappings)`
- `mimic_2_omop  匹配集 (47 mappings)`

这样可以把 `SMD` 数据集分开用于训练和匹配，而不是只保留一份。

## 训练

运行：

```bash
python smart_train.py
```

流程：

1. 选择数据集
2. 系统自动分析数据特征
3. 显示推荐流程、推荐策略、推荐模型
4. 由 ReAct 智能体调用训练工具
5. 保存模型到 [trained_models](/d:/my_design/trained_models)
6. 保存训练结果到 [training_results_improved.json](/d:/my_design/training_results_improved.json)

## 匹配

运行：

```bash
python smart_match.py
```

流程：

1. 选择数据集
2. 系统自动分析数据特征
3. 由 ReAct 智能体调用匹配工具
4. 输出结果到对应的 `matching_results_*.json`
5. 控制台打印评估指标

当前匹配结束后会打印这些指标：

- `Accuracy`
- `Top1 Accuracy`
- `Precision`
- `Recall`
- `F1`

## 结果文件

- 训练结果：[training_results_improved.json](/d:/my_design/training_results_improved.json)
- Beer 匹配结果：[matching_results_Beer.json](/d:/my_design/matching_results_Beer.json)
- DBLP-ACM 匹配结果：[matching_results_DBLP-ACM.json](/d:/my_design/matching_results_DBLP-ACM.json)
- MIMIC_2_OMOP 匹配结果：[matching_results_mimic_2_omop.json](/d:/my_design/matching_results_mimic_2_omop.json)

## 当前推荐用法

- 想看训练效果：先运行 `python smart_train.py`
- 想得到最终匹配：再运行 `python smart_match.py`

如果模型不存在，匹配流程会自动补训练。

## 说明

这份文档对应的是当前代码，而不是早期的旧版菜单系统。
如果后面你还要，我可以继续把控制台里的“推荐流程 / 数据特征类别”文案再改得更自然一点。
