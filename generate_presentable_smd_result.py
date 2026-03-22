import json
from pathlib import Path


def main() -> None:
    src = Path("matching_results_mimic_2_omop.json")
    out = Path("matching_results_mimic_2_omop_presentable.json")
    raw = json.loads(src.read_text(encoding="utf-8"))

    best_by_source = {}
    for match in raw.get("matches", []):
        source_id = str(match.get("source_id", "")).strip()
        if not source_id:
            continue
        score = float(match.get("ranking_score", match.get("confidence", 0.0)))
        current = best_by_source.get(source_id)
        current_score = (
            float(current.get("ranking_score", current.get("confidence", 0.0)))
            if current is not None
            else -1.0
        )
        if score > current_score:
            best_by_source[source_id] = match

    presentable_matches = []
    for source_id in sorted(best_by_source):
        match = best_by_source[source_id]
        presentable_matches.append(
            {
                "source_id": match.get("source_id"),
                "source_table": match.get("source_table"),
                "source_column": match.get("source_column"),
                "target_id": match.get("target_id"),
                "target_table": match.get("target_table"),
                "target_column": match.get("target_column"),
                "confidence": round(float(match.get("confidence", 0.0)), 6),
                "ranking_score": round(
                    float(match.get("ranking_score", match.get("confidence", 0.0))), 6
                ),
                "match_method": match.get("match_method"),
                "gold_label": int(match.get("gold_label", 0)),
                "expected_targets": match.get("expected_targets", []),
                "review_required": bool(match.get("review_required", False)),
            }
        )

    evaluation = raw.get("evaluation", {})
    report = {
        "dataset": raw.get("dataset"),
        "data_type": "SMD",
        "view_type": "presentable_top1_view",
        "match_count": len(presentable_matches),
        "model_type": raw.get("model_type"),
        "threshold": raw.get("threshold"),
        "top_k_per_source": raw.get("top_k_per_source"),
        "scene": raw.get("scene"),
        "matching_strategy": raw.get("matching_strategy"),
        "scene_detector": raw.get("scene_detector"),
        "scene_confidence": raw.get("scene_confidence"),
        "summary": {
            "evaluated_source_count": raw.get("evaluated_source_count"),
            "holdout_pair_count": raw.get("holdout_pair_count"),
            "top1_accuracy": evaluation.get("top1_accuracy"),
            "top3_accuracy": evaluation.get("top3_accuracy"),
            "top5_accuracy": evaluation.get("top5_accuracy"),
            "mrr": evaluation.get("mrr"),
            "candidate_precision": evaluation.get("candidate_precision"),
            "llm_reranked_source_count": raw.get("llm_reranked_source_count"),
            "note": "展示版结果：每个源字段仅保留排序后的最佳候选，适合汇报与展示。",
        },
        "agent_diagnostics": {
            "issues": raw.get("agent_diagnostics", {}).get("issues", []),
            "summary": raw.get("agent_diagnostics", {}).get("summary", ""),
        },
        "matches": presentable_matches,
    }

    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
