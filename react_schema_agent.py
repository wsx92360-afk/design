"""LangChain tool-calling agent for schema matching workflow orchestration."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_ollama import ChatOllama

from train_improved_schema_matching import ImprovedSchemaMatchingTrainer

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langgraph").setLevel(logging.WARNING)


@dataclass
class AgentRunResult:
    success: bool
    workflow_backend: str
    scene_info: Optional[Dict]
    payload: Optional[Dict]
    final_message: str
    action_log: List[str] = field(default_factory=list)
    failure_reason: str = ""


class ReActSchemaAgent:
    """Use LangChain tool-calling to dynamically select matching actions."""

    def __init__(self, trainer: ImprovedSchemaMatchingTrainer) -> None:
        self.trainer = trainer
        self.model_name = trainer.scene_identifier._ollama_model
        self.base_url = trainer.scene_identifier._ollama_host
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0,
        )
        self._latest_scene_info: Optional[Dict] = None
        self._latest_payload: Optional[Dict] = None
        self._current_match_config: Dict[str, object] = {}
        self._action_log: List[str] = []
        self._selected_toolchain: str = ""
        self._tool_selection_timeout_sec = float(os.environ.get("AGENT_TOOL_SELECTION_TIMEOUT_SECONDS", "35"))
        self._tool_selection_retry_timeout_sec = float(os.environ.get("AGENT_TOOL_SELECTION_RETRY_TIMEOUT_SECONDS", "55"))
        self._enable_final_smd_llm_rerank = os.environ.get("ENABLE_FINAL_SMD_LLM_RERANK", "0").strip().lower() in {"1", "true", "yes"}

    def _log_action(self, message: str) -> None:
        self._action_log.append(message)
        print(f"[Agent] {message}")

    @staticmethod
    def _to_bool(value: object) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def _apply_toolchain_config(
        self,
        toolchain_name: str,
        *,
        threshold: float,
        normalize_before_match: bool,
        top_k_per_source: int,
        enable_llm_rerank: bool,
    ) -> Dict[str, object]:
        self._selected_toolchain = toolchain_name
        self._current_match_config.update({
            "threshold": float(threshold),
            "normalize_before_match": bool(normalize_before_match),
            "top_k_per_source": int(top_k_per_source),
            "enable_llm_rerank": bool(enable_llm_rerank),
            "selected_toolchain": toolchain_name,
        })
        self._log_action(
            f"选择工具链 `{toolchain_name}`，"
            f"threshold={float(threshold):.2f}，"
            f"normalize_before_match={bool(normalize_before_match)}，"
            f"top_k_per_source={int(top_k_per_source)}，"
            f"enable_llm_rerank={bool(enable_llm_rerank)}"
        )
        return self._current_match_config.copy()

    def _fallback_select_toolchain(self, scene_info: Optional[Dict]) -> Dict[str, object]:
        scene_type = (scene_info or {}).get("type", "")
        quality_label = (scene_info or {}).get("quality_label", "")
        if scene_type == "SMD":
            return self._apply_toolchain_config(
                "smd_semantic_balanced",
                threshold=0.46,
                normalize_before_match=False,
                top_k_per_source=1,
                enable_llm_rerank=False,
            )
        if quality_label == "dirty_like":
            return self._apply_toolchain_config(
                "sld_dirty_clean_first",
                threshold=0.50,
                normalize_before_match=True,
                top_k_per_source=1,
                enable_llm_rerank=False,
            )
        if quality_label == "structured_like":
            return self._apply_toolchain_config(
                "sld_structured_fast",
                threshold=0.50,
                normalize_before_match=False,
                top_k_per_source=1,
                enable_llm_rerank=False,
            )
        return self._apply_toolchain_config(
            "sld_textual_balanced",
            threshold=0.48,
            normalize_before_match=False,
            top_k_per_source=1,
            enable_llm_rerank=False,
        )

    def _run_match_once(
        self,
        dataset_name: str,
        split_role: str,
        threshold: float,
        normalize_before_match: bool,
        top_k_per_source: int,
        enable_llm_rerank: bool,
    ) -> Dict:
        result = self.trainer.match_single_dataset(
            dataset_name,
            threshold=threshold,
            split_role=split_role,
            normalize_before_match=normalize_before_match,
            top_k_per_source=top_k_per_source,
            enable_llm_rerank=enable_llm_rerank,
        )
        self._current_match_config = {
            "dataset_name": dataset_name,
            "split_role": split_role,
            "threshold": float(threshold),
            "normalize_before_match": bool(normalize_before_match),
            "top_k_per_source": int(top_k_per_source),
            "enable_llm_rerank": bool(enable_llm_rerank),
        }
        self._latest_payload = result
        return result or {"error": "matching failed"}

    def _should_run_final_smd_llm_rerank(self, payload: Optional[Dict]) -> bool:
        if not payload:
            return False
        if payload.get("data_type") != "SMD":
            return False
        if not self._enable_final_smd_llm_rerank:
            return False
        if bool(payload.get("enable_llm_rerank", False)):
            return False

        match_count = int(payload.get("match_count", 0))
        evaluated_sources = int(payload.get("evaluated_source_count", 0))
        diagnostics = self._analyze_match_state(payload, self._latest_scene_info)
        evaluation = payload.get("evaluation", {}) or {}
        top1_accuracy = float(evaluation.get("top1_accuracy", evaluation.get("accuracy", 0.0)))
        blocking_issues = {
            "too_many_candidates",
            "one_to_many_conflict",
        }
        if any(issue in blocking_issues for issue in diagnostics.get("issues", [])):
            return False
        if match_count <= 0:
            return False
        if evaluated_sources and match_count > evaluated_sources:
            return False
        if top1_accuracy < 0.25:
            return True
        if "field_type_incompatible" in diagnostics.get("issues", []):
            return True
        if "target_fields_uncovered" in diagnostics.get("issues", []):
            return True
        return match_count <= min(20, max(8, evaluated_sources))

    def _analyze_match_state(self, payload: Optional[Dict], scene_info: Optional[Dict]) -> Dict:
        if not payload:
            return {
                "issues": ["match_failed"],
                "summary": "匹配未生成结果。",
                "suggested_actions": ["run_match"],
            }

        matches = payload.get("matches", []) or []
        evaluation = payload.get("evaluation", {}) or {}
        threshold = float(payload.get("threshold", self._current_match_config.get("threshold", 0.5)))
        issues = []
        notes = []

        match_count = int(payload.get("match_count", len(matches)))
        if match_count == 0:
            if threshold >= 0.3:
                issues.append("zero_candidates_after_filtering")
                notes.append("过滤后候选对为零，当前阈值可能偏严。")
            else:
                issues.append("no_matches")
                notes.append("当前没有找到任何可接受匹配。")

        if payload.get("data_type") == "SMD":
            evaluated = int(payload.get("evaluated_source_count", 0))
            if evaluated and match_count > max(evaluated * 2, 20):
                issues.append("too_many_candidates")
                notes.append("候选结果数量偏多，当前阈值可能偏松。")
        else:
            evaluated_pairs = int(evaluation.get("evaluated_pair_count", 0))
            if evaluated_pairs and match_count > max(int(evaluated_pairs * 0.5), 50):
                issues.append("too_many_candidates")
                notes.append("预测为匹配的候选过多，建议提高过滤阈值。")

        source_key_name = "source_id" if payload.get("data_type") == "SMD" else "ltable_id"
        source_counter = Counter(str(match.get(source_key_name, "")) for match in matches if match.get(source_key_name) is not None)
        conflicted_sources = [key for key, count in source_counter.items() if key and count > 1]
        if conflicted_sources:
            issues.append("one_to_many_conflict")
            notes.append(f"检测到 {len(conflicted_sources)} 个源字段或记录存在一对多冲突。")

        type_conflicts = [
            match for match in matches
            if str(match.get("source_type", "")).strip()
            and str(match.get("target_type", "")).strip()
            and str(match.get("source_type", "")).strip().lower() != str(match.get("target_type", "")).strip().lower()
        ]
        if type_conflicts:
            issues.append("field_type_incompatible")
            notes.append(f"检测到 {len(type_conflicts)} 条匹配存在字段类型不兼容。")

        ambiguous_matches = [
            match for match in matches
            if 0.4 <= float(match.get("confidence", 0.0)) <= 0.6
        ]
        if ambiguous_matches:
            issues.append("ambiguous_results")
            notes.append(f"检测到 {len(ambiguous_matches)} 条结果落在模糊置信度区间。")

        if scene_info and scene_info.get("quality_label") == "dirty_like":
            issues.append("data_quality_risk")
            notes.append("内容分析显示当前数据集存在较明显的脏数据风险，可能影响结果。")

        uncovered_targets: List[str] = []
        if payload.get("data_type") == "SMD":
            expected_targets = set()
            matched_targets = set()
            for match in matches:
                matched_targets.add(str(match.get("target_id", "")))
                for target_id in match.get("expected_targets", []) or []:
                    expected_targets.add(str(target_id))
            uncovered_targets = sorted(target for target in expected_targets if target and target not in matched_targets)
            if uncovered_targets:
                issues.append("target_fields_uncovered")
                notes.append(f"仍有 {len(uncovered_targets)} 个目标字段未被覆盖。")

        suggested_actions = []
        if "zero_candidates_after_filtering" in issues:
            suggested_actions.append("lower_threshold_and_retry")
        if "too_many_candidates" in issues:
            suggested_actions.append("raise_threshold_and_retry")
        if "data_quality_risk" in issues:
            suggested_actions.append("clean_and_retry")
        if "one_to_many_conflict" in issues:
            suggested_actions.append("resolve_one_to_many_conflicts")
        suggested_actions.append("finalize_report")

        return {
            "issues": issues,
            "summary": "；".join(notes) if notes else "未检测到明显异常，结果可直接整理输出。",
            "suggested_actions": suggested_actions,
            "stats": {
                "match_count": match_count,
                "threshold": threshold,
                "ambiguous_count": len(ambiguous_matches),
                "type_conflict_count": len(type_conflicts),
                "one_to_many_conflict_count": len(conflicted_sources),
                "uncovered_target_count": len(uncovered_targets),
            },
            "uncovered_targets": uncovered_targets[:20],
        }

    def _resolve_one_to_many_conflicts(self, payload: Optional[Dict]) -> Dict:
        if not payload:
            return {"error": "no current payload"}

        matches = payload.get("matches", []) or []
        source_key_name = "source_id" if payload.get("data_type") == "SMD" else "ltable_id"
        best_by_source: Dict[str, Dict] = {}
        dropped = 0

        for match in sorted(matches, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
            source_key = str(match.get(source_key_name, ""))
            if source_key and source_key not in best_by_source:
                best_by_source[source_key] = match
            elif source_key:
                dropped += 1
            else:
                best_by_source[f"__orphan__{len(best_by_source)}"] = match

        payload["matches"] = list(best_by_source.values())
        payload["match_count"] = len(payload["matches"])
        payload.setdefault("agent_actions", []).append({
            "action": "resolve_one_to_many_conflicts",
            "dropped_conflicting_matches": dropped,
        })
        self._latest_payload = payload
        return {
            "resolved_match_count": len(payload["matches"]),
            "dropped_conflicting_matches": dropped,
        }

    def _apply_manual_review_flags(self, payload: Optional[Dict], low: float = 0.4, high: float = 0.6) -> Dict:
        if not payload:
            return {"error": "no current payload"}

        flagged = 0
        for match in payload.get("matches", []) or []:
            confidence = float(match.get("confidence", 0.0))
            if low <= confidence <= high:
                match["review_required"] = True
                match["review_reason"] = "confidence_in_ambiguous_band"
                flagged += 1

        payload.setdefault("agent_actions", []).append({
            "action": "flag_ambiguous_matches",
            "flagged_count": flagged,
        })
        self._latest_payload = payload
        return {"flagged_for_manual_review": flagged}

    def _finalize_payload(self, payload: Optional[Dict], scene_info: Optional[Dict]) -> Optional[Dict]:
        if payload is None:
            return None

        diagnostics = self._analyze_match_state(payload, scene_info)
        self._apply_manual_review_flags(payload)
        payload["agent_diagnostics"] = diagnostics
        payload["agent_summary"] = diagnostics.get("summary", "")
        strategy_trace = [
            action.get("action")
            for action in payload.get("agent_actions", [])
            if isinstance(action, dict) and action.get("action")
        ]
        payload["agent_strategy_trace"] = strategy_trace
        if self._selected_toolchain:
            payload["selected_toolchain"] = self._selected_toolchain

        issue_to_strategy = {
            "zero_candidates_after_filtering": "lower_threshold_and_retry",
            "too_many_candidates": "raise_threshold_and_retry",
            "field_type_incompatible": "warn_type_conflict",
            "one_to_many_conflict": "resolve_one_to_many_conflicts",
            "no_matches": "report_no_matches",
            "target_fields_uncovered": "report_uncovered_targets",
            "data_quality_risk": "clean_and_retry",
            "ambiguous_results": "flag_ambiguous_matches",
        }
        payload["agent_issue_strategy_map"] = [
            {
                "issue": issue,
                "strategy": issue_to_strategy.get(issue, "report_only"),
            }
            for issue in diagnostics.get("issues", [])
        ]
        return payload

    def _build_execution_tools(self, task: str) -> Dict[str, BaseTool]:
        trainer = self.trainer
        agent = self

        @tool
        def train_dataset(dataset_name: str, split_role: str = "train") -> str:
            """Train the selected dataset."""
            result = trainer.train_single_dataset(dataset_name, split_role=split_role)
            trainer.save_results()
            agent._latest_payload = result
            agent._log_action(f"调用工具 `train_dataset`，数据集={dataset_name}，split_role={split_role}")
            return json.dumps(result or {"error": "training failed"}, ensure_ascii=False)

        @tool
        def use_smd_semantic_balanced_toolchain() -> str:
            """Select the balanced SMD semantic matching toolchain."""
            config = agent._apply_toolchain_config(
                "smd_semantic_balanced",
                threshold=0.46,
                normalize_before_match=False,
                top_k_per_source=1,
                enable_llm_rerank=False,
            )
            return json.dumps(config, ensure_ascii=False)

        @tool
        def use_smd_semantic_expanded_toolchain() -> str:
            """Select the expanded SMD semantic retrieval toolchain."""
            config = agent._apply_toolchain_config(
                "smd_semantic_expanded",
                threshold=0.44,
                normalize_before_match=False,
                top_k_per_source=3,
                enable_llm_rerank=False,
            )
            return json.dumps(config, ensure_ascii=False)

        @tool
        def use_smd_precise_llm_toolchain() -> str:
            """Select the precise SMD semantic adjudication toolchain with LLM rerank enabled."""
            config = agent._apply_toolchain_config(
                "smd_precise_llm",
                threshold=0.48,
                normalize_before_match=False,
                top_k_per_source=1,
                enable_llm_rerank=True,
            )
            return json.dumps(config, ensure_ascii=False)

        @tool
        def use_sld_structured_fast_toolchain() -> str:
            """Select the fast structured SLD toolchain."""
            config = agent._apply_toolchain_config(
                "sld_structured_fast",
                threshold=0.50,
                normalize_before_match=False,
                top_k_per_source=1,
                enable_llm_rerank=False,
            )
            return json.dumps(config, ensure_ascii=False)

        @tool
        def use_sld_textual_balanced_toolchain() -> str:
            """Select the balanced textual SLD toolchain."""
            config = agent._apply_toolchain_config(
                "sld_textual_balanced",
                threshold=0.48,
                normalize_before_match=False,
                top_k_per_source=1,
                enable_llm_rerank=False,
            )
            return json.dumps(config, ensure_ascii=False)

        @tool
        def use_sld_dirty_clean_first_toolchain() -> str:
            """Select the dirty-data-first SLD toolchain with normalization enabled."""
            config = agent._apply_toolchain_config(
                "sld_dirty_clean_first",
                threshold=0.50,
                normalize_before_match=True,
                top_k_per_source=1,
                enable_llm_rerank=False,
            )
            return json.dumps(config, ensure_ascii=False)

        @tool
        def run_match(
            dataset_name: str,
            split_role: str = "match",
            threshold: str = "0.5",
            normalize_before_match: str = "false",
            top_k_per_source: str = "1",
            enable_llm_rerank: str = "true",
        ) -> str:
            """Run matching with the provided controls."""
            result = agent._run_match_once(
                dataset_name=dataset_name,
                split_role=split_role,
                threshold=float(threshold),
                normalize_before_match=agent._to_bool(normalize_before_match),
                top_k_per_source=int(top_k_per_source),
                enable_llm_rerank=agent._to_bool(enable_llm_rerank),
            )
            agent._log_action(
                "调用工具 `run_match`，"
                f"threshold={float(threshold):.2f}，"
                f"normalize_before_match={agent._to_bool(normalize_before_match)}，"
                f"top_k_per_source={int(top_k_per_source)}，"
                f"enable_llm_rerank={agent._to_bool(enable_llm_rerank)}，"
                f"match_count={result.get('match_count', 0)}"
            )
            return json.dumps(result, ensure_ascii=False)

        @tool
        def inspect_match_state() -> str:
            """Inspect the latest matching result and return current issues."""
            report = agent._analyze_match_state(agent._latest_payload, agent._latest_scene_info)
            agent._log_action(
                "调用工具 `inspect_match_state`，"
                f"issues={','.join(report.get('issues', [])) or 'none'}"
            )
            return json.dumps(report, ensure_ascii=False)

        @tool
        def lower_threshold_and_retry(step: str = "0.1") -> str:
            """Lower the current threshold and rerun matching."""
            current = float(agent._current_match_config.get("threshold", 0.5))
            next_threshold = max(0.05, current - float(step))
            cfg = agent._current_match_config
            result = agent._run_match_once(
                dataset_name=str(cfg["dataset_name"]),
                split_role=str(cfg["split_role"]),
                threshold=next_threshold,
                normalize_before_match=bool(cfg.get("normalize_before_match", False)),
                top_k_per_source=int(cfg.get("top_k_per_source", 1)),
                enable_llm_rerank=bool(cfg.get("enable_llm_rerank", True)),
            )
            agent._log_action(
                f"调用工具 `lower_threshold_and_retry`，新阈值={next_threshold:.2f}，match_count={result.get('match_count', 0)}"
            )
            return json.dumps({"new_threshold": next_threshold, "match_count": result.get("match_count", 0)}, ensure_ascii=False)

        @tool
        def raise_threshold_and_retry(step: str = "0.1") -> str:
            """Raise the current threshold and rerun matching."""
            current = float(agent._current_match_config.get("threshold", 0.5))
            next_threshold = min(0.95, current + float(step))
            cfg = agent._current_match_config
            result = agent._run_match_once(
                dataset_name=str(cfg["dataset_name"]),
                split_role=str(cfg["split_role"]),
                threshold=next_threshold,
                normalize_before_match=bool(cfg.get("normalize_before_match", False)),
                top_k_per_source=int(cfg.get("top_k_per_source", 1)),
                enable_llm_rerank=bool(cfg.get("enable_llm_rerank", True)),
            )
            agent._log_action(
                f"调用工具 `raise_threshold_and_retry`，新阈值={next_threshold:.2f}，match_count={result.get('match_count', 0)}"
            )
            return json.dumps({"new_threshold": next_threshold, "match_count": result.get("match_count", 0)}, ensure_ascii=False)

        @tool
        def clean_and_retry() -> str:
            """Normalize dirty textual data and rerun matching."""
            cfg = agent._current_match_config
            result = agent._run_match_once(
                dataset_name=str(cfg["dataset_name"]),
                split_role=str(cfg["split_role"]),
                threshold=float(cfg.get("threshold", 0.5)),
                normalize_before_match=True,
                top_k_per_source=int(cfg.get("top_k_per_source", 1)),
                enable_llm_rerank=bool(cfg.get("enable_llm_rerank", True)),
            )
            agent._log_action(
                f"调用工具 `clean_and_retry`，normalize_before_match=True，match_count={result.get('match_count', 0)}"
            )
            return json.dumps({"normalize_before_match": True, "match_count": result.get("match_count", 0)}, ensure_ascii=False)

        @tool
        def resolve_one_to_many_conflicts() -> str:
            """Keep only the highest-confidence match per source item."""
            result = agent._resolve_one_to_many_conflicts(agent._latest_payload)
            agent._log_action(
                "调用工具 `resolve_one_to_many_conflicts`，"
                f"dropped={result.get('dropped_conflicting_matches', 0)}"
            )
            return json.dumps(result, ensure_ascii=False)

        @tool
        def finalize_report() -> str:
            """Attach diagnostics and manual-review flags to the current payload."""
            payload = agent._finalize_payload(agent._latest_payload, agent._latest_scene_info)
            agent._latest_payload = payload
            if payload is None:
                return json.dumps({"error": "no current payload"}, ensure_ascii=False)
            diagnostics = payload.get("agent_diagnostics", {})
            agent._log_action(
                "调用工具 `finalize_report`，"
                f"issues={','.join(diagnostics.get('issues', [])) or 'none'}"
            )
            return json.dumps(diagnostics, ensure_ascii=False)

        if task == "train":
            return {train_dataset.name: train_dataset}
        return {
            use_smd_semantic_balanced_toolchain.name: use_smd_semantic_balanced_toolchain,
            use_smd_semantic_expanded_toolchain.name: use_smd_semantic_expanded_toolchain,
            use_smd_precise_llm_toolchain.name: use_smd_precise_llm_toolchain,
            use_sld_structured_fast_toolchain.name: use_sld_structured_fast_toolchain,
            use_sld_textual_balanced_toolchain.name: use_sld_textual_balanced_toolchain,
            use_sld_dirty_clean_first_toolchain.name: use_sld_dirty_clean_first_toolchain,
            run_match.name: run_match,
            inspect_match_state.name: inspect_match_state,
            lower_threshold_and_retry.name: lower_threshold_and_retry,
            raise_threshold_and_retry.name: raise_threshold_and_retry,
            clean_and_retry.name: clean_and_retry,
            resolve_one_to_many_conflicts.name: resolve_one_to_many_conflicts,
            finalize_report.name: finalize_report,
        }

    def _invoke_tool_loop(
        self,
        system_prompt: str,
        tools: Dict[str, BaseTool],
        max_steps: int = 3,
        step_timeout_sec: Optional[float] = None,
    ) -> str:
        llm_with_tools = self.llm.bind_tools(list(tools.values()))
        messages = [SystemMessage(content=system_prompt), HumanMessage(content="开始执行。")]
        final_message = ""
        timeout_sec = float(step_timeout_sec or self._tool_selection_timeout_sec)

        for step in range(max_steps):
            self._log_action(f"[Step {step + 1}] 正在请求模型决定下一步工具...")
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(llm_with_tools.invoke, messages)
                    ai_message = future.result(timeout=timeout_sec)
            except FuturesTimeoutError as exc:
                raise TimeoutError("agent tool-selection llm timeout") from exc
            messages.append(ai_message)
            tool_calls = getattr(ai_message, "tool_calls", None) or []

            if not tool_calls:
                content = getattr(ai_message, "content", "")
                final_message = str(content or "").strip()
                break

            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                args = tool_call.get("args", {}) or {}
                tool_id = tool_call.get("id", "")
                if tool_name not in tools:
                    observation = json.dumps({"error": f"unknown tool: {tool_name}"}, ensure_ascii=False)
                    messages.append(ToolMessage(content=observation, tool_call_id=tool_id, name=tool_name))
                    continue

                self._log_action(f"[Step {step + 1}] 模型选择工具 `{tool_name}`")
                observation = tools[tool_name].invoke(args)
                messages.append(
                    ToolMessage(
                        content=str(observation),
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )

        return final_message

    def _select_toolchain_with_llm_name(
        self,
        system_prompt: str,
        allowed_tool_names: List[str],
        timeout_sec: float,
    ) -> str:
        prompt = (
            f"{system_prompt}\n\n"
            "Return exactly one tool name from this list and nothing else:\n"
            + "\n".join(allowed_tool_names)
        )
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.llm.invoke, [HumanMessage(content=prompt)])
                message = future.result(timeout=timeout_sec)
            content = str(getattr(message, "content", "") or "").strip()
            for tool_name in allowed_tool_names:
                if tool_name in content:
                    return tool_name
            normalized = content.strip().splitlines()[0].strip("` ").strip()
            if normalized in allowed_tool_names:
                return normalized
        except Exception:
            return ""
        return ""

    def _plan_match_toolchain(self, dataset_name: str, split_role: str, scene_info: Optional[Dict]) -> None:
        tools = self._build_execution_tools("match")
        all_toolchain_tools = {
            name: tool
            for name, tool in tools.items()
            if name.startswith("use_smd_") or name.startswith("use_sld_")
        }
        if not all_toolchain_tools:
            self._fallback_select_toolchain(scene_info)
            return

        scene_type = (scene_info or {}).get("type", "")
        quality_label = (scene_info or {}).get("quality_label", "")
        if scene_type == "SMD":
            allowed_names = [
                "use_smd_semantic_balanced_toolchain",
                "use_smd_semantic_expanded_toolchain",
                "use_smd_precise_llm_toolchain",
            ]
        elif quality_label == "dirty_like":
            allowed_names = [
                "use_sld_dirty_clean_first_toolchain",
                "use_sld_textual_balanced_toolchain",
                "use_sld_structured_fast_toolchain",
            ]
        elif quality_label == "structured_like":
            allowed_names = [
                "use_sld_structured_fast_toolchain",
                "use_sld_textual_balanced_toolchain",
            ]
        else:
            allowed_names = [
                "use_sld_textual_balanced_toolchain",
                "use_sld_structured_fast_toolchain",
            ]

        toolchain_tools = {
            name: all_toolchain_tools[name]
            for name in allowed_names
            if name in all_toolchain_tools
        }
        if not toolchain_tools:
            self._fallback_select_toolchain(scene_info)
            return

        system_prompt = f"""
You are a schema matching agent selecting one toolchain.
data_type={scene_info.get('type', '') if scene_info else ''}
quality_label={scene_info.get('quality_label', '') if scene_info else ''}

Rules:
- SMD usually choose use_smd_semantic_balanced_toolchain.
- Choose use_smd_precise_llm_toolchain only if stricter semantic adjudication is needed.
- SLD dirty_like choose use_sld_dirty_clean_first_toolchain.
- SLD structured_like choose use_sld_structured_fast_toolchain.
- SLD textual_like choose use_sld_textual_balanced_toolchain.
""".strip()

        chosen_tool_name = self._select_toolchain_with_llm_name(
            system_prompt,
            list(toolchain_tools.keys()),
            timeout_sec=self._tool_selection_retry_timeout_sec,
        )
        if chosen_tool_name and chosen_tool_name in toolchain_tools:
            self._log_action(f"[Step 1] 模型选择工具 `{chosen_tool_name}`")
            toolchain_tools[chosen_tool_name].invoke({})
        else:
            self._log_action("智能体未返回有效工具链选择结果，切换到本地快速兜底选择。")
            self._fallback_select_toolchain(scene_info)

    def _bootstrap_match_workflow(self, dataset_name: str, split_role: str, scene_info: Optional[Dict]) -> Dict:
        self._log_action("正在执行基础匹配。")
        bootstrap_result = self._run_match_once(
            dataset_name=dataset_name,
            split_role=split_role,
            threshold=float(self._current_match_config.get("threshold", 0.5)),
            normalize_before_match=bool(self._current_match_config.get("normalize_before_match", False)),
            top_k_per_source=int(self._current_match_config.get("top_k_per_source", 1)),
            enable_llm_rerank=bool(self._current_match_config.get("enable_llm_rerank", False)),
        )
        self._log_action("预执行基础匹配完成。")
        diagnostics = self._analyze_match_state(bootstrap_result, scene_info)
        self._log_action(
            "预执行诊断完成，"
            f"issues={','.join(diagnostics.get('issues', [])) or 'none'}"
        )
        return diagnostics

    def _run_fast_path_actions(self, diagnostics: Dict, scene_info: Optional[Dict]) -> bool:
        issues = set(diagnostics.get("issues", []))
        applied = False

        if "data_quality_risk" in issues:
            self._log_action("采用快速策略：检测到脏数据风险，直接执行清洗重试。")
            self._build_execution_tools("match")["clean_and_retry"].invoke({})
            diagnostics = self._analyze_match_state(self._latest_payload, scene_info)
            issues = set(diagnostics.get("issues", []))
            applied = True

        if "zero_candidates_after_filtering" in issues:
            self._log_action("采用快速策略：候选为零，直接降低阈值重试。")
            self._build_execution_tools("match")["lower_threshold_and_retry"].invoke({"step": "0.1"})
            diagnostics = self._analyze_match_state(self._latest_payload, scene_info)
            issues = set(diagnostics.get("issues", []))
            applied = True

        if "too_many_candidates" in issues:
            self._log_action("采用快速策略：候选过多，直接提高阈值重试。")
            self._current_match_config["top_k_per_source"] = 1
            self._build_execution_tools("match")["raise_threshold_and_retry"].invoke({"step": "0.1"})
            diagnostics = self._analyze_match_state(self._latest_payload, scene_info)
            issues = set(diagnostics.get("issues", []))
            applied = True

        if "one_to_many_conflict" in issues:
            self._log_action("采用快速策略：检测到一对多冲突，直接执行消歧。")
            self._build_execution_tools("match")["resolve_one_to_many_conflicts"].invoke({})
            applied = True

        if applied:
            self._build_execution_tools("match")["finalize_report"].invoke({})

        return applied

    def run(
        self,
        task: str,
        dataset_name: str,
        split_role: str = "default",
        scene_info: Optional[Dict] = None,
    ) -> AgentRunResult:
        self._latest_scene_info = scene_info
        self._latest_payload = None
        self._action_log = []
        self._selected_toolchain = ""
        default_top_k = 1 if scene_info and scene_info.get("type") == "SMD" else 1
        self._current_match_config = {
            "dataset_name": dataset_name,
            "split_role": split_role,
            "threshold": 0.5,
            "normalize_before_match": False,
            "top_k_per_source": default_top_k,
            "enable_llm_rerank": False,
        }

        tools = self._build_execution_tools(task)
        failure_reason = ""

        if task == "train":
            system_prompt = f"""
You are a schema matching training agent.

Known context:
- dataset_name: {dataset_name}
- split_role: {split_role}
- recommended_model: {scene_info.get('recommended_model', '') if scene_info else ''}

You must call the `train_dataset` tool exactly once.
After the tool call, respond with a short summary.
""".strip()
        else:
            self._log_action("开始由智能体选择场景适配的工具链。")
            self._plan_match_toolchain(dataset_name, split_role, scene_info)
            diagnostics = self._bootstrap_match_workflow(dataset_name, split_role, scene_info)
            if self._run_fast_path_actions(diagnostics, scene_info):
                if self._should_run_final_smd_llm_rerank(self._latest_payload):
                    self._log_action("快速纠偏完成，当前候选规模较小，补充执行最终 LLM 精判重排。")
                    cfg = self._current_match_config
                    self._run_match_once(
                        dataset_name=str(cfg["dataset_name"]),
                        split_role=str(cfg["split_role"]),
                        threshold=float(cfg.get("threshold", 0.5)),
                        normalize_before_match=bool(cfg.get("normalize_before_match", False)),
                        top_k_per_source=int(cfg.get("top_k_per_source", 1)),
                        enable_llm_rerank=True,
                    )
                elif scene_info and scene_info.get("type") == "SMD":
                    self._log_action("快速纠偏完成，已跳过最终 LLM 精判重排，避免覆盖纠偏结果并减少耗时。")
                payload = self._finalize_payload(self._latest_payload, scene_info)
                return AgentRunResult(
                    success=payload is not None,
                    workflow_backend=f"langchain_fastpath_agent:{self.model_name}",
                    scene_info=scene_info,
                    payload=payload,
                    final_message="Fast-path corrective actions applied.",
                    action_log=list(self._action_log),
                    failure_reason="",
                )
            actionable = [
                issue for issue in diagnostics.get("issues", [])
                if issue in {
                    "zero_candidates_after_filtering",
                    "too_many_candidates",
                    "one_to_many_conflict",
                    "data_quality_risk",
                }
            ]
            if not actionable:
                self._log_action("未检测到需要纠偏的可执行问题，直接整理报告。")
                if (
                    scene_info
                    and scene_info.get("type") == "SMD"
                    and diagnostics.get("issues")
                    and any(issue in {"field_type_incompatible", "target_fields_uncovered", "ambiguous_results"} for issue in diagnostics.get("issues", []))
                    and self._enable_final_smd_llm_rerank
                ):
                    self._log_action("检测到 SMD 排序质量仍不足，切换到精判强化模式重新执行。")
                    self._apply_toolchain_config(
                        "smd_precise_llm",
                        threshold=0.46,
                        normalize_before_match=False,
                        top_k_per_source=1,
                        enable_llm_rerank=True,
                    )
                    cfg = self._current_match_config
                    self._run_match_once(
                        dataset_name=str(cfg["dataset_name"]),
                        split_role=str(cfg["split_role"]),
                        threshold=float(cfg.get("threshold", 0.46)),
                        normalize_before_match=bool(cfg.get("normalize_before_match", False)),
                        top_k_per_source=int(cfg.get("top_k_per_source", 1)),
                        enable_llm_rerank=True,
                    )
                elif self._should_run_final_smd_llm_rerank(self._latest_payload):
                    self._log_action("当前结果稳定，候选规模较小，补充执行最终 LLM 精判重排。")
                    cfg = self._current_match_config
                    self._run_match_once(
                        dataset_name=str(cfg["dataset_name"]),
                        split_role=str(cfg["split_role"]),
                        threshold=float(cfg.get("threshold", 0.5)),
                        normalize_before_match=bool(cfg.get("normalize_before_match", False)),
                        top_k_per_source=int(cfg.get("top_k_per_source", 1)),
                        enable_llm_rerank=True,
                    )
                elif scene_info and scene_info.get("type") == "SMD":
                    self._log_action("当前结果稳定，已跳过最终 LLM 精判重排以控制耗时。")
                payload = self._finalize_payload(self._latest_payload, scene_info)
                return AgentRunResult(
                    success=payload is not None,
                    workflow_backend=f"langchain_tools_agent:{self.model_name}",
                    scene_info=scene_info,
                    payload=payload,
                    final_message="No corrective action needed.",
                    action_log=list(self._action_log),
                    failure_reason="",
                )

            system_prompt = f"""
You are a schema matching correction agent.

Known context:
- dataset_name: {dataset_name}
- split_role: {split_role}
- scene: {scene_info.get('scene', '') if scene_info else ''}
- data_type: {scene_info.get('type', '') if scene_info else ''}
- matching_strategy: {scene_info.get('matching_strategy', '') if scene_info else ''}

Current diagnostics:
{json.dumps(diagnostics, ensure_ascii=False)}

Only use tools to correct actionable issues.
Rules:
- zero_candidates_after_filtering -> prefer `lower_threshold_and_retry`
- too_many_candidates -> prefer `raise_threshold_and_retry`
- data_quality_risk -> prefer `clean_and_retry`
- one_to_many_conflict -> prefer `resolve_one_to_many_conflicts`
- call `inspect_match_state` after a corrective action if needed
- always call `finalize_report` before finishing

Keep the workflow short and efficient.
""".strip()

        try:
            max_steps = 1 if task == "train" else 3
            final_message = self._invoke_tool_loop(system_prompt, tools, max_steps=max_steps)
        except Exception as exc:
            final_message = ""
            failure_reason = str(exc)

        payload = self._latest_payload
        if payload is None:
            failure_reason = failure_reason or "模型未产生有效工具调用或未生成结果。"
            self._log_action("模型未顺利完成工具调用，切换到本地兜底流程。")
            if task == "train":
                payload = self.trainer.train_single_dataset(dataset_name, split_role=split_role)
                if payload is not None:
                    self.trainer.save_results()
            else:
                payload = self._finalize_payload(
                    self.trainer.match_single_dataset(dataset_name, split_role=split_role),
                    scene_info,
                )
        else:
            if task == "match":
                payload = self._finalize_payload(payload, scene_info)

        return AgentRunResult(
            success=payload is not None,
            workflow_backend=f"langchain_tools_agent:{self.model_name}",
            scene_info=scene_info,
            payload=payload,
            final_message=final_message,
            action_log=list(self._action_log),
            failure_reason=failure_reason,
        )
