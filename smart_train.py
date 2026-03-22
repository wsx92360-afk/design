"""
统一智能训练入口。

用途:
- 展示所有数据集
- 用户选择数据集后自动识别场景
- 自动选择对应模型与训练策略
"""

import time

from react_schema_agent import ReActSchemaAgent
from train_improved_schema_matching import ImprovedSchemaMatchingTrainer


def main() -> None:
    total_start = time.perf_counter()
    trainer = ImprovedSchemaMatchingTrainer('datasets')
    agent = ReActSchemaAgent(trainer)
    dataset_list = trainer.build_dataset_selection_menu(expand_smd_splits=True)
    if not dataset_list:
        raise SystemExit(1)

    print("\n输入数据集编号后，系统会自动分析数据特征并选择对应模型。")
    print(f"当前智能体框架: LangChain + ReAct")
    print(f"当前大模型后端: {agent.model_name}")
    choice = input("请选择数据集编号: ").strip()

    if not choice.isdigit():
        raise SystemExit("请输入有效编号。")

    index = int(choice)
    if index < 1 or index > len(dataset_list):
        raise SystemExit("编号超出范围。")

    selected = dataset_list[index - 1]
    dataset_name = selected["dataset_name"]
    split_role = selected.get("split_role", "default")
    print("\n[1/4] 已选择数据集")
    print("[2/4] 正在用非智能体场景识别器分析数据集特征，请稍候...")
    scene_start = time.perf_counter()
    scene_info = trainer.identify_dataset_scene(dataset_name)
    if scene_info is None:
        raise SystemExit("未能识别该数据集。")
    scene_elapsed = time.perf_counter() - scene_start

    print("\n[3/4] 场景识别完成")
    print("分析结果:")
    print(f"  数据集: {selected['display_name']}")
    print(f"  推荐流程: {scene_info['scene']}")
    print(f"  数据特征类别: {scene_info['type']}")
    print(f"  推荐策略: {scene_info['matching_strategy']}")
    print(f"  推荐模型: {scene_info['recommended_model']}")
    print(f"  识别器: {scene_info['scene_detector']}")
    print(f"  置信度: {scene_info['scene_confidence']:.2f}")
    print(f"  依据: {scene_info['scene_evidence']}")
    if scene_info.get("quality_label"):
        print(f"  质量画像: {scene_info['quality_label']}")
    if scene_info.get("quality_evidence"):
        print(f"  质量依据: {scene_info['quality_evidence']}")
    print(f"  分析耗时: {scene_elapsed:.2f}s")

    confirm = input("\n继续训练? (y/n, 默认为 y): ").strip().lower() or "y"
    if confirm != "y":
        raise SystemExit(0)

    print("\n[4/4] 正在调用 LangChain + ReAct 智能体选择并执行训练方法，请稍候...")
    exec_start = time.perf_counter()
    run_result = agent.run("train", dataset_name, split_role=split_role, scene_info=scene_info)
    if not run_result.success or run_result.payload is None:
        raise SystemExit("训练失败。")
    exec_elapsed = time.perf_counter() - exec_start
    total_elapsed = time.perf_counter() - total_start

    payload = run_result.payload
    print("\n[OK] 训练完成")
    print(f"  智能体框架: LangChain + ReAct")
    print(f"  工作流后端: {run_result.workflow_backend}")
    if run_result.failure_reason:
        print(f"  工具调用异常: {run_result.failure_reason}")
    if run_result.action_log:
        print("  执行流程:")
        for action in run_result.action_log:
            print(f"    - {action}")
    strategy_trace = payload.get("agent_strategy_trace", [])
    if strategy_trace:
        print("  采用策略:")
        for strategy in strategy_trace:
            print(f"    - {strategy}")
    issue_strategy_map = payload.get("agent_issue_strategy_map", [])
    if issue_strategy_map:
        print("  问题与策略:")
        for item in issue_strategy_map:
            print(f"    - 问题: {item.get('issue', 'unknown')}")
            print(f"      策略: {item.get('strategy', 'report_only')}")
    if payload.get("dataset_name"):
        print(f"  数据集: {payload['dataset_name']}")
    if payload.get("model_type"):
        print(f"  模型: {payload['model_type']}")
    evaluation = payload.get("evaluation", {})
    if evaluation:
        if "accuracy" in evaluation:
            print(f"  Accuracy: {evaluation['accuracy']:.4f}")
        if "precision" in evaluation:
            print(f"  Precision: {evaluation['precision']:.4f}")
        if "recall" in evaluation:
            print(f"  Recall: {evaluation['recall']:.4f}")
        if "f1_score" in evaluation:
            print(f"  F1: {evaluation['f1_score']:.4f}")
    print(f"  执行耗时: {exec_elapsed:.2f}s")
    print(f"  总耗时: {total_elapsed:.2f}s")


if __name__ == '__main__':
    main()
