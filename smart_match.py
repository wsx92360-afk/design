"""统一智能匹配入口。"""

from __future__ import annotations

import json
import time
from pathlib import Path

from react_schema_agent import ReActSchemaAgent
from train_improved_schema_matching import ImprovedSchemaMatchingTrainer


def main() -> None:
    total_start = time.perf_counter()
    trainer = ImprovedSchemaMatchingTrainer("datasets")
    agent = ReActSchemaAgent(trainer)
    dataset_list = trainer.build_dataset_selection_menu(expand_smd_splits=True)
    if not dataset_list:
        raise SystemExit(1)

    print("\n输入数据集编号后，系统会先调用 llama3.1 分析数据特征，再选择匹配方法。")
    print("当前智能体框架: LangChain + ReAct")
    print(f"当前大模型后端: {trainer.scene_identifier._ollama_model}")
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
        raise SystemExit("未能识别数据集场景。")
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

    print("\n[4/4] 正在调用 LangChain + ReAct 智能体选择并执行匹配方法，请稍候...")
    exec_start = time.perf_counter()
    run_result = agent.run("match", dataset_name, split_role=split_role, scene_info=scene_info)
    result = run_result.payload
    if result is None:
        raise SystemExit("匹配失败。")
    exec_elapsed = time.perf_counter() - exec_start
    total_elapsed = time.perf_counter() - total_start

    output_path = Path(f"matching_results_{dataset_name}.json")
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] 匹配完成，结果已保存到: {output_path}")
    print(f"[OK] 智能体框架: LangChain + ReAct")
    print(f"[OK] 工作流后端: {run_result.workflow_backend}")
    if run_result.failure_reason:
        print(f"[Agent] 工具调用异常: {run_result.failure_reason}")
    if run_result.action_log:
        print("[Agent] 执行流程:")
        for action in run_result.action_log:
            print(f"  - {action}")
    if result.get("selected_toolchain"):
        print(f"[Agent] 工具链: {result['selected_toolchain']}")
    strategy_trace = result.get("agent_strategy_trace", [])
    if strategy_trace:
        print("[Agent] 采用策略:")
        for strategy in strategy_trace:
            print(f"  - {strategy}")
    print(f"[OK] 匹配数: {result['match_count']}")
    if "llm_reranked_source_count" in result:
        print(f"[OK] LLM 精判重排源字段数: {result['llm_reranked_source_count']}")
    if "llm_validated_source_count" in result:
        print(f"[OK] LLM 最终验证源字段数: {result['llm_validated_source_count']}")
    diagnostics = result.get("agent_diagnostics", {})
    if diagnostics:
        print("[Agent] 动态诊断:")
        issues = diagnostics.get("issues", [])
        print(f"  问题标签: {', '.join(issues) if issues else 'none'}")
        if diagnostics.get("summary"):
            print(f"  诊断摘要: {diagnostics['summary']}")
    issue_strategy_map = result.get("agent_issue_strategy_map", [])
    if issue_strategy_map:
        print("[Agent] 问题与策略:")
        for item in issue_strategy_map:
            print(f"  - 问题: {item.get('issue', 'unknown')}")
            print(f"    策略: {item.get('strategy', 'report_only')}")
    evaluation = result.get("evaluation", {})
    if evaluation:
        print("[Metrics] 评估指标:")
        if evaluation.get("metric_family") == "ranking":
            if "accuracy" in evaluation:
                print(f"  Accuracy: {evaluation['accuracy']:.4f}")
            if "precision" in evaluation:
                print(f"  Precision: {evaluation['precision']:.4f}")
            if "recall" in evaluation:
                print(f"  Recall: {evaluation['recall']:.4f}")
            if "f1_score" in evaluation:
                print(f"  F1: {evaluation['f1_score']:.4f}")
            if "top1_accuracy" in evaluation:
                print(f"  Top1 Accuracy: {evaluation['top1_accuracy']:.4f}")
            if "top3_accuracy" in evaluation:
                print(f"  Top3 Accuracy: {evaluation['top3_accuracy']:.4f}")
            if "top5_accuracy" in evaluation:
                print(f"  Top5 Accuracy: {evaluation['top5_accuracy']:.4f}")
            if "mrr" in evaluation:
                print(f"  MRR: {evaluation['mrr']:.4f}")
            if "candidate_precision" in evaluation:
                print(f"  Candidate Precision: {evaluation['candidate_precision']:.4f}")
        elif "accuracy" in evaluation:
            print(f"  Accuracy: {evaluation['accuracy']:.4f}")
        if evaluation.get("metric_family") != "ranking" and "precision" in evaluation:
            print(f"  Precision: {evaluation['precision']:.4f}")
        if evaluation.get("metric_family") != "ranking" and "recall" in evaluation:
            print(f"  Recall: {evaluation['recall']:.4f}")
        if evaluation.get("metric_family") != "ranking" and "f1_score" in evaluation:
            print(f"  F1: {evaluation['f1_score']:.4f}")
    matches = result.get("matches", []) or []
    if matches:
        print("[Preview] 匹配样例:")
        preview_count = min(5, len(matches))
        for match in matches[:preview_count]:
            if result.get("data_type") == "SMD":
                source_label = f"{match.get('source_table', '')}.{match.get('source_column', '')}"
                target_label = f"{match.get('target_table', '')}.{match.get('target_column', '')}"
            else:
                source_label = str(match.get("ltable_id", ""))
                target_label = str(match.get("rtable_id", ""))
            gold_label = match.get("gold_label")
            gold_text = f", gold={gold_label}" if gold_label is not None else ""
            review_text = ", review" if match.get("review_required") else ""
            print(
                f"  - {source_label} -> {target_label} | "
                f"confidence={float(match.get('confidence', 0.0)):.4f}{gold_text}{review_text}"
            )
    print(f"[Time] 执行耗时: {exec_elapsed:.2f}s")
    print(f"[Time] 总耗时: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
