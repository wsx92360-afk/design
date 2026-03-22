"""
基于数据集内容摘要的场景识别器。

优先级:
1. 本地 HuggingFace 文本分类/零样本模型（如果用户已在本机准备好模型）
2. 内容启发式回退（不依赖路径名）

说明:
- 这里的识别依据是数据集文件结构、字段摘要、样本内容、gold mapping 等“内容信号”
- 不使用数据集路径名来直接判断 SMD / SLD
"""

from __future__ import annotations

import json
import os
import requests
from requests.exceptions import Timeout as RequestsTimeout
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SceneDecision:
    scene: str
    data_type: str
    matching_strategy: str
    recommended_model: str
    detector: str
    confidence: float
    evidence: str
    quality_label: str = ""
    quality_evidence: str = ""


class SceneIdentifier:
    """根据数据集内容自动识别 SMD / SLD 场景。"""

    def __init__(self) -> None:
        self._classifier = None
        self._backend_name = None
        self._ollama_model = os.environ.get("SCENE_OLLAMA_MODEL", "llama3.1:latest").strip()
        self._ollama_host = os.environ.get("SCENE_OLLAMA_HOST", "http://127.0.0.1:11434").strip().rstrip("/")
        self._ollama_timeout = int(os.environ.get("SCENE_OLLAMA_TIMEOUT_SECONDS", "20"))
        self._try_init_local_hf_classifier()

    def _try_init_local_hf_classifier(self) -> None:
        model_path = os.environ.get("SCENE_LLM_MODEL_PATH", "").strip()
        if not model_path:
            return

        try:
            from transformers import pipeline

            self._classifier = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path,
            )
            self._backend_name = f"local_huggingface:{model_path}"
        except Exception:
            self._classifier = None
            self._backend_name = None

    def build_summary(self, dataset_name: str, dataset_payload: Dict) -> str:
        payload = {
            "dataset_name": dataset_name,
            "has_gold_mapping": dataset_payload.get("has_gold_mapping", False),
            "source_count": dataset_payload.get("source_count", 0),
            "row_counts": dataset_payload.get("row_counts", {}),
            "column_counts": dataset_payload.get("column_counts", {}),
            "sample_columns": dataset_payload.get("sample_columns", {}),
            "file_types": dataset_payload.get("file_types", []),
            "has_instance_values": dataset_payload.get("has_instance_values", False),
            "non_empty_ratio": dataset_payload.get("non_empty_ratio", 0.0),
            "avg_text_length": dataset_payload.get("avg_text_length", 0.0),
            "noisy_text_ratio": dataset_payload.get("noisy_text_ratio", 0.0),
            "dirty_pattern_ratio": dataset_payload.get("dirty_pattern_ratio", 0.0),
            "metadata_like_schema": dataset_payload.get("metadata_like_schema", False),
            "dataset_notes": dataset_payload.get("dataset_notes", ""),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _infer_quality_label(dataset_payload: Dict, data_type: str) -> tuple[str, str]:
        if data_type == "SMD":
            return "metadata_only", "仅检测到元数据级 schema 信号。"

        noisy_text_ratio = float(dataset_payload.get("noisy_text_ratio", 0.0))
        dirty_pattern_ratio = float(dataset_payload.get("dirty_pattern_ratio", 0.0))
        avg_text_length = float(dataset_payload.get("avg_text_length", 0.0))
        non_empty_ratio = float(dataset_payload.get("non_empty_ratio", 0.0))
        empty_ratio = max(0.0, 1.0 - non_empty_ratio)

        if (
            noisy_text_ratio >= 0.10
            or empty_ratio >= 0.35
            or (18.0 <= avg_text_length <= 30.0 and dirty_pattern_ratio >= 0.22)
        ):
            return "dirty_like", "样本值中存在较多噪声、混合格式或缺失情况，呈现脏数据特征。"
        if avg_text_length >= 30.0 and noisy_text_ratio < 0.08:
            return "textual_like", "样本文本较长且噪声相对较低，更接近文本型场景。"
        if avg_text_length >= 28.0:
            return "textual_like", "样本文本较长且描述性较强，更接近文本型场景。"
        return "structured_like", "样本值较规整且文本长度较短，更接近结构化场景。"

    def identify(self, dataset_name: str, dataset_payload: Dict) -> SceneDecision:
        summary = self.build_summary(dataset_name, dataset_payload)

        heuristic_decision = self._identify_with_content_heuristic(dataset_payload)
        if self._should_short_circuit_with_heuristic(dataset_payload, heuristic_decision):
            return heuristic_decision

        decision = self._identify_with_ollama(summary, dataset_payload)
        if decision is not None:
            return decision

        if self._classifier is not None:
            decision = self._identify_with_local_classifier(summary, dataset_payload)
            if decision is not None:
                return decision

        return heuristic_decision

    @staticmethod
    def _should_short_circuit_with_heuristic(
        dataset_payload: Dict,
        heuristic_decision: SceneDecision,
    ) -> bool:
        has_gold_mapping = bool(dataset_payload.get("has_gold_mapping", False))
        has_instance_values = bool(dataset_payload.get("has_instance_values", False))
        metadata_like_schema = bool(dataset_payload.get("metadata_like_schema", False))
        file_types = set(dataset_payload.get("file_types", []))
        source_count = int(dataset_payload.get("source_count", 0))
        row_counts = dataset_payload.get("row_counts", {}) or {}

        if (
            heuristic_decision.data_type == "SMD"
            and has_gold_mapping
            and (metadata_like_schema or "xml" in file_types or source_count >= 2)
        ):
            return True

        table_files = {
            str(name).lower()
            for name in row_counts.keys()
            if str(name).lower().endswith(".csv")
        }
        has_sld_pair_files = {"tablea.csv", "tableb.csv"} <= table_files
        if (
            heuristic_decision.data_type == "SLD"
            and has_instance_values
            and has_sld_pair_files
        ):
            return True

        return False

    def _parse_model_decision(self, parsed: Dict, dataset_payload: Dict, detector: str) -> Optional[SceneDecision]:
        data_type = str(parsed.get("data_type", "")).upper()
        if data_type not in {"SLD", "SMD"}:
            return None

        quality_label, quality_evidence = self._infer_quality_label(dataset_payload, data_type)
        if data_type == "SMD":
            recommended_model = "gradient_boosting"
        else:
            recommended_model = (
                "random_forest"
                if quality_label == "structured_like"
                else "gradient_boosting"
            )

        scene = str(parsed.get("scene", "")).strip()
        if scene not in {"schema_with_instance_data", "schema_only_matching"}:
            scene = "schema_with_instance_data" if data_type == "SLD" else "schema_only_matching"

        matching_strategy = str(parsed.get("matching_strategy", "")).strip()
        if matching_strategy not in {"sld_instance_aware_matching", "smd_field_level_matching"}:
            matching_strategy = "sld_instance_aware_matching" if data_type == "SLD" else "smd_field_level_matching"

        confidence = float(parsed.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        evidence = str(parsed.get("evidence", "")).strip() or "模型根据数据集内容摘要完成场景识别。"

        return SceneDecision(
            scene=scene,
            data_type=data_type,
            matching_strategy=matching_strategy,
            recommended_model=recommended_model,
            detector=detector,
            confidence=confidence,
            evidence=evidence,
            quality_label=quality_label,
            quality_evidence=quality_evidence,
        )

    def _identify_with_ollama(self, summary: str, dataset_payload: Dict) -> Optional[SceneDecision]:
        prompt = f"""
You are a schema matching scene classifier.
Decide whether the dataset below is:
- SLD: schema matching with instance data
- SMD: schema matching with only metadata

Return strict JSON with keys:
data_type, scene, matching_strategy, recommended_model, confidence, evidence

Rules:
- data_type must be either "SLD" or "SMD"
- scene must be either "schema_with_instance_data" or "schema_only_matching"
- matching_strategy must be either "sld_instance_aware_matching" or "smd_field_level_matching"
- recommended_model must be either "random_forest" or "gradient_boosting"
- confidence must be a number between 0 and 1
- evidence must be one short sentence

Dataset summary:
{summary}
""".strip()

        try:
            response = requests.post(
                f"{self._ollama_host}/api/generate",
                json={
                    "model": self._ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0,
                    },
                },
                timeout=self._ollama_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            raw_text = payload.get("response", "").strip()
            if not raw_text:
                return None
            parsed = json.loads(raw_text)
            return self._parse_model_decision(
                parsed,
                dataset_payload,
                f"ollama:{self._ollama_model}",
            )
        except RequestsTimeout:
            return None
        except Exception:
            return None

    def _identify_with_local_classifier(self, summary: str, dataset_payload: Dict) -> Optional[SceneDecision]:
        try:
            result = self._classifier(summary)
            if not result:
                return None
            top = result[0]
            label = str(top.get("label", "")).upper()
            score = float(top.get("score", 0.0))

            if "SMD" in label:
                quality_label, quality_evidence = self._infer_quality_label(dataset_payload, "SMD")
                return SceneDecision(
                    scene="schema_only_matching",
                    data_type="SMD",
                    matching_strategy="smd_field_level_matching",
                    recommended_model="gradient_boosting",
                    detector=self._backend_name or "local_huggingface",
                    confidence=score,
                    evidence="本地 HuggingFace 模型根据数据集内容摘要判定为 schema-only 场景。",
                    quality_label=quality_label,
                    quality_evidence=quality_evidence,
                )

            if "SLD" in label:
                quality_label, quality_evidence = self._infer_quality_label(dataset_payload, "SLD")
                recommended_model = (
                    "random_forest"
                    if quality_label == "structured_like"
                    else "gradient_boosting"
                )
                return SceneDecision(
                    scene="schema_with_instance_data",
                    data_type="SLD",
                    matching_strategy="sld_instance_aware_matching",
                    recommended_model=recommended_model,
                    detector=self._backend_name or "local_huggingface",
                    confidence=score,
                    evidence="本地 HuggingFace 模型根据数据集内容摘要判定为 instance-aware 场景。",
                    quality_label=quality_label,
                    quality_evidence=quality_evidence,
                )
        except Exception:
            return None

        return None

    def _identify_with_content_heuristic(self, dataset_payload: Dict) -> SceneDecision:
        has_gold_mapping = dataset_payload.get("has_gold_mapping", False)
        has_instance_values = dataset_payload.get("has_instance_values", False)
        non_empty_ratio = float(dataset_payload.get("non_empty_ratio", 0.0))
        source_count = int(dataset_payload.get("source_count", 0))
        metadata_like_schema = bool(dataset_payload.get("metadata_like_schema", False))
        file_types = set(dataset_payload.get("file_types", []))
        smd_quality_label, smd_quality_evidence = self._infer_quality_label(dataset_payload, "SMD")
        sld_quality_label, sld_quality_evidence = self._infer_quality_label(dataset_payload, "SLD")

        if has_gold_mapping and (metadata_like_schema or "xml" in file_types or source_count >= 2) and not has_instance_values:
            return SceneDecision(
                scene="schema_only_matching",
                data_type="SMD",
                matching_strategy="smd_field_level_matching",
                recommended_model="gradient_boosting",
                detector="content_heuristic_fallback",
                confidence=0.88,
                evidence="检测到 gold mapping 与 schema 级结构信号，且未发现可靠实例值证据，因此判定为 SMD。",
                quality_label=smd_quality_label,
                quality_evidence=smd_quality_evidence,
            )

        if has_gold_mapping and metadata_like_schema:
            return SceneDecision(
                scene="schema_only_matching",
                data_type="SMD",
                matching_strategy="smd_field_level_matching",
                recommended_model="gradient_boosting",
                detector="content_heuristic_fallback",
                confidence=0.84,
                evidence="检测到 gold mapping 和元数据型字段描述列，因此优先判定为 SMD。",
                quality_label=smd_quality_label,
                quality_evidence=smd_quality_evidence,
            )

        if has_instance_values or non_empty_ratio > 0.3:
            recommended_model = (
                "random_forest"
                if sld_quality_label == "structured_like"
                else "gradient_boosting"
            )
            return SceneDecision(
                scene="schema_with_instance_data",
                data_type="SLD",
                matching_strategy="sld_instance_aware_matching",
                recommended_model=recommended_model,
                detector="content_heuristic_fallback",
                confidence=0.76,
                evidence="检测到实例值密度较高或明确的实例数据表，因此判定为 SLD。",
                quality_label=sld_quality_label,
                quality_evidence=sld_quality_evidence,
            )

        return SceneDecision(
            scene="schema_only_matching",
            data_type="SMD",
            matching_strategy="smd_field_level_matching",
            recommended_model="gradient_boosting",
            detector="content_heuristic_fallback",
            confidence=0.60,
            evidence="未检测到充足实例值信号，默认归入 schema-only 场景。",
            quality_label=smd_quality_label,
            quality_evidence=smd_quality_evidence,
        )
