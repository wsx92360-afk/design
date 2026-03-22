"""
改进的 Schema Matching 训练器
融合最新论文的多种特征相似度计算方法

论文参考：
1. LLMatch (2507.10897) - 多阶段优化策略
2. Matchmaker (2410.24105) - 候选生成与细化
3. KG-RAG4SM (2501.08686) - 知识图谱增强
4. Schema Matching with LLMs (2407.11852) - 上下文感知
5. Scalable Schema Mapping (2505.24716) - 可扩展性策略
"""

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from typing import Dict, Optional, List, Tuple
import json
import pickle
from pathlib import Path
from dataset_loader import SchemaMatchingDataset, DataPreprocessor
from smd_dataset_loader import SMDDatasetLoader, extract_smd_features
from scene_identifier import SceneIdentifier
import logging
from contextlib import suppress
import re
from sklearn.model_selection import train_test_split as sk_train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AdvancedSimilarityMetrics:
    """高级相似度度量类 - 融合多种方法"""
    
    @staticmethod
    def string_similarity(s1: str, s2: str) -> float:
        """基于 SequenceMatcher 的字符串相似度"""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    @staticmethod
    def jaccard_similarity(s1: str, s2: str, ngram_size: int = 2) -> float:
        """N-gram 级别的 Jaccard 相似度"""
        def get_ngrams(text, n):
            text = text.lower()
            return set([text[i:i+n] for i in range(len(text)-n+1)])
        
        ngrams1 = get_ngrams(s1, ngram_size)
        ngrams2 = get_ngrams(s2, ngram_size)
        
        if len(ngrams1 | ngrams2) == 0:
            return 0.0
        
        return len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> float:
        """标准化的编辑距离"""
        s1, s2 = s1.lower(), s2.lower()
        
        if max(len(s1), len(s2)) == 0:
            return 1.0
        
        # 动态规划计算编辑距离
        dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        
        for i in range(len(s1) + 1):
            dp[i][0] = i
        for j in range(len(s2) + 1):
            dp[0][j] = j
        
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        # 标准化到 [0, 1]
        return 1.0 - (dp[len(s1)][len(s2)] / max(len(s1), len(s2)))
    
    @staticmethod
    def token_overlap(s1: str, s2: str) -> float:
        """基于词汇重叠的相似度"""
        tokens1 = set(s1.lower().split())
        tokens2 = set(s2.lower().split())
        
        if len(tokens1 | tokens2) == 0:
            return 0.0
        
        return len(tokens1 & tokens2) / len(tokens1 | tokens2)
    
    @staticmethod
    def contextual_similarity(val1: str, val2: str, 
                            contains_threshold: float = 0.8) -> float:
        """上下文相似度 - 考虑包含关系"""
        v1, v2 = str(val1).lower(), str(val2).lower()
        
        # 如果一个包含另一个
        if v1 in v2 and len(v1) > 0:
            return len(v1) / len(v2) if len(v2) > 0 else 1.0
        elif v2 in v1 and len(v2) > 0:
            return len(v2) / len(v1) if len(v1) > 0 else 1.0
        
        return 0.0
    
    @staticmethod
    def phonetic_similarity(s1: str, s2: str) -> float:
        """基于首字符和长度的音韵相似度（简化版）"""
        s1, s2 = str(s1).lower(), str(s2).lower()
        
        # 首字符相同性
        first_char_match = 1.0 if s1 and s2 and s1[0] == s2[0] else 0.0
        
        # 长度相似性
        len_max = max(len(s1), len(s2))
        len_sim = 1.0 - abs(len(s1) - len(s2)) / len_max if len_max > 0 else 0.0
        
        return 0.6 * first_char_match + 0.4 * len_sim
    
    @staticmethod
    def data_type_compatibility(val1: str, val2: str) -> float:
        """数据类型兼容性检查"""
        def infer_type(val):
            val = str(val).strip()
            
            # 尝试转换为数字
            try:
                float(val)
                return 'numeric'
            except:
                pass
            
            # 检查是否为日期格式
            if any(x in val for x in ['-', '/', '年', '月', '日']):
                return 'date'
            
            # 检查是否为布尔值
            if val.lower() in ['true', 'false', 'yes', 'no', '是', '否']:
                return 'boolean'
            
            # 默认为文本
            return 'text'
        
        type1 = infer_type(val1)
        type2 = infer_type(val2)
        
        return 1.0 if type1 == type2 else 0.3

class ImprovedSchemaMatchingTrainer:
    """改进的 Schema Matching 训练器"""
    
    def __init__(self, dataset_dir: str = 'datasets'):
        self.dataset_loader = SchemaMatchingDataset(dataset_dir)
        self.smd_loader = SMDDatasetLoader(dataset_dir + '/SMD')
        self.models = {}
        self.results = {}
        self.similarity_metrics = AdvancedSimilarityMetrics()
        self.scene_identifier = SceneIdentifier()
        self.model_dir = Path('trained_models')
        self.model_dir.mkdir(exist_ok=True)
        self._menu_dataset_cache: Dict[str, Dict] = {}
        self._scene_info_cache: Dict[str, Dict] = {}
        self._smd_llm_rerank_cache: Dict[str, Optional[str]] = {}
        self._smd_embedding_vectorizer = HashingVectorizer(
            n_features=512,
            alternate_sign=False,
            norm="l2",
            analyzer="char_wb",
            ngram_range=(3, 5),
        )
        self._smd_llm_validation_cache: Dict[str, Optional[Dict]] = {}
        self._train_examples_cache: Dict[str, List[Dict]] = {}
        self._current_match_context: Dict[str, str] = {}
        self._smd_prior_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
        alias_cache_path = Path("mimic_field_aliases.json")
        if alias_cache_path.exists():
            with suppress(Exception):
                self._field_alias_cache = json.loads(alias_cache_path.read_text(encoding="utf-8"))
            if not hasattr(self, "_field_alias_cache"):
                self._field_alias_cache = {}
        else:
            self._field_alias_cache = {}

    def _model_artifact_path(self, dataset_name: str) -> Path:
        safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', dataset_name)
        return self.model_dir / f"{safe_name}.pkl"

    def _smd_dataset_path(self, dataset_name: str) -> Path:
        datasets = self.smd_loader.list_datasets()
        if dataset_name not in datasets:
            raise FileNotFoundError(f"未找到 SMD 数据集: {dataset_name}")
        return datasets[dataset_name]

    def _peek_smd_split_counts(self, dataset_name: str) -> Dict[str, int]:
        dataset_path = self._smd_dataset_path(dataset_name)
        counts = {'gold_total': 0, 'train_count': 0, 'match_count': 0}

        for file_name, key in [
            ('gold_mapping.csv', 'gold_total'),
            ('gold_mapping_train.csv', 'train_count'),
            ('gold_mapping_match.csv', 'match_count'),
        ]:
            file_path = dataset_path / file_name
            if not file_path.exists():
                continue
            with suppress(Exception):
                counts[key] = max(sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1, 0)

        return counts

    def _locate_dataset_path(self, dataset_name: str) -> Optional[Dict[str, str]]:
        smd_datasets = self.smd_loader.list_datasets()
        if dataset_name in smd_datasets:
            return {
                'path': str(smd_datasets[dataset_name]),
                'locator': 'datasets/SMD',
            }

        for category in ['Structured', 'Textual', 'Dirty']:
            dataset_path = self._resolve_sld_dataset_path(category, dataset_name)
            if dataset_path.exists():
                return {
                    'path': str(dataset_path),
                    'locator': f'datasets/{category}',
                }

        return None

    def _resolve_sld_dataset_path(self, category: str, dataset_name: str) -> Path:
        category_dir = Path(self.dataset_loader.dataset_dir) / category
        nested_dir = category_dir / category
        if nested_dir.exists():
            return nested_dir / dataset_name
        return category_dir / dataset_name

    def _build_lightweight_sld_scene_payload(self, category: str, dataset_name: str) -> Dict:
        dataset_path = self._resolve_sld_dataset_path(category, dataset_name)
        if not dataset_path.exists():
            return {}

        row_counts = {}
        column_counts = {}
        sample_columns = {}
        non_empty_cells = 0
        total_cells = 0
        file_types = set()
        sample_text_lengths: List[float] = []
        noisy_text_values = 0
        total_text_values = 0

        def _looks_noisy(text: str) -> bool:
            if not text:
                return False
            special_count = sum(1 for ch in text if not ch.isalnum() and ch not in {" ", "_", "-", "/", "."})
            upper_count = sum(1 for ch in text if ch.isupper())
            return (
                "  " in text
                or special_count >= 3
                or (len(text) >= 8 and upper_count >= max(4, len(text) // 2))
            )

        csv_files = sorted(dataset_path.glob('*.csv'))
        for csv_file in csv_files:
            try:
                sample_df = pd.read_csv(csv_file, nrows=50)
                row_count = 0
                with open(csv_file, 'r', encoding='utf-8', errors='ignore') as handle:
                    row_count = max(sum(1 for _ in handle) - 1, 0)

                file_name = csv_file.name
                row_counts[file_name] = int(row_count)
                column_counts[file_name] = int(len(sample_df.columns))
                sample_columns[file_name] = [str(col) for col in list(sample_df.columns)[:8]]
                total_cells += int(sample_df.shape[0] * sample_df.shape[1])
                non_empty_cells += int(sample_df.notna().sum().sum())
                object_df = sample_df.select_dtypes(include=['object'])
                if not object_df.empty:
                    text_values = [
                        str(value).strip()
                        for value in object_df.astype(str).stack().tolist()
                        if str(value).strip() and str(value).strip().lower() != 'nan'
                    ]
                    if text_values:
                        sample_text_lengths.append(float(np.mean([len(value) for value in text_values])))
                        total_text_values += len(text_values)
                        noisy_text_values += sum(1 for value in text_values if _looks_noisy(value))
                file_types.add(csv_file.suffix.lstrip('.').lower())
            except Exception:
                continue

        non_empty_ratio = (non_empty_cells / total_cells) if total_cells > 0 else 0.0
        has_instance_values = any(
            rows > 0 and column_counts.get(file_name, 0) > 0
            for file_name, rows in row_counts.items()
            if file_name.lower().startswith(('tablea', 'tableb'))
        )
        avg_text_length = float(np.mean(sample_text_lengths)) if sample_text_lengths else 0.0
        noisy_text_ratio = (noisy_text_values / total_text_values) if total_text_values > 0 else 0.0

        return {
            'has_gold_mapping': False,
            'source_count': len(row_counts),
            'row_counts': row_counts,
            'column_counts': column_counts,
            'sample_columns': sample_columns,
            'file_types': sorted(file_types) if file_types else ['csv'],
            'has_instance_values': has_instance_values,
            'non_empty_ratio': non_empty_ratio,
            'dataset_notes': 'entity matching tables with lightweight sampled rows',
            'avg_text_length': avg_text_length,
            'noisy_text_ratio': noisy_text_ratio,
        }

    def _build_content_only_scene_payload(self, dataset_name: str) -> Dict:
        located = self._locate_dataset_path(dataset_name)
        if located is None:
            return {}

        dataset_path = Path(located['path'])
        row_counts: Dict[str, int] = {}
        column_counts: Dict[str, int] = {}
        sample_columns: Dict[str, List[str]] = {}
        file_types = set()
        total_cells = 0
        non_empty_cells = 0
        sample_text_lengths: List[float] = []
        noisy_text_values = 0
        total_text_values = 0
        dirty_pattern_values = 0
        metadata_like_schema = False
        schema_like_signals = 0
        schema_like_columns = {
            'tablename',
            'columnname',
            'columntype',
            'columndesc',
            'ispk',
            'isfk',
            'source_table',
            'source_column',
            'target_table',
            'target_column',
        }

        gold_mapping_files = list(dataset_path.rglob('gold_mapping*.csv'))
        xml_files = list(dataset_path.rglob('*.xml'))
        csv_files = [path for path in dataset_path.rglob('*.csv') if path not in gold_mapping_files]

        for xml_file in xml_files[:20]:
            try:
                source_name = xml_file.parent.name
                field_names, records = self.smd_loader.parse_xml(str(xml_file), verbose=False)
                row_counts[source_name] = int(len(records))
                column_counts[source_name] = int(len(field_names))
                sample_columns[source_name] = [str(col) for col in field_names[:8]]
                file_types.add('xml')
            except Exception:
                continue

        for csv_file in csv_files[:20]:
            try:
                sample_df = pd.read_csv(csv_file, nrows=50)
                with open(csv_file, 'r', encoding='utf-8', errors='ignore') as handle:
                    row_count = max(sum(1 for _ in handle) - 1, 0)

                file_name = csv_file.name
                row_counts[file_name] = int(row_count)
                column_counts[file_name] = int(len(sample_df.columns))
                sample_columns[file_name] = [str(col) for col in list(sample_df.columns)[:8]]
                normalized_columns = {
                    re.sub(r'[^a-z0-9]+', '', str(col).lower())
                    for col in sample_df.columns
                }
                overlap_count = len(normalized_columns & schema_like_columns)
                if overlap_count >= 3:
                    schema_like_signals += 1
                total_cells += int(sample_df.shape[0] * sample_df.shape[1])
                non_empty_cells += int(sample_df.notna().sum().sum())
                object_df = sample_df.select_dtypes(include=['object'])
                if not object_df.empty:
                    text_values = [
                        str(value).strip()
                        for value in object_df.astype(str).stack().tolist()
                        if str(value).strip() and str(value).strip().lower() != 'nan'
                    ]
                    if text_values:
                        sample_text_lengths.append(float(np.mean([len(value) for value in text_values])))
                        total_text_values += len(text_values)
                        noisy_text_values += sum(
                            1
                            for value in text_values
                            if (
                                "  " in value
                                or sum(1 for ch in value if not ch.isalnum() and ch not in {" ", "_", "-", "/", "."}) >= 3
                                or (
                                    len(value) >= 8
                                    and sum(1 for ch in value if ch.isupper()) >= max(4, len(value) // 2)
                                )
                            )
                        )
                        dirty_pattern_values += sum(
                            1
                            for value in text_values
                            if (
                                len(value) >= 12
                                and (
                                    ("(" in value and ")" in value)
                                    or ("," in value and len(value.split(",")) >= 2)
                                    or ("/" in value and len(value.split("/")) >= 2)
                                    or (":" in value)
                                    or (
                                        any(ch.isalpha() for ch in value)
                                        and any(ch.isdigit() for ch in value)
                                    )
                                )
                            )
                        )
                file_types.add('csv')
            except Exception:
                continue

        has_gold_mapping = bool(gold_mapping_files)
        metadata_like_schema = bool(xml_files) or schema_like_signals > 0
        non_empty_ratio = (non_empty_cells / total_cells) if total_cells > 0 else 0.0
        has_instance_values = total_cells > 0 and non_empty_ratio > 0.05 and not metadata_like_schema
        source_count = len(row_counts)
        avg_text_length = float(np.mean(sample_text_lengths)) if sample_text_lengths else 0.0
        noisy_text_ratio = (noisy_text_values / total_text_values) if total_text_values > 0 else 0.0
        dirty_pattern_ratio = (dirty_pattern_values / total_text_values) if total_text_values > 0 else 0.0

        notes = []
        if has_gold_mapping:
            notes.append('gold mapping detected')
        if xml_files:
            notes.append('xml schema sources detected')
        if csv_files:
            notes.append('csv tables detected')
        if metadata_like_schema:
            notes.append('metadata-like schema columns detected')

        return {
            'has_gold_mapping': has_gold_mapping,
            'source_count': source_count,
            'row_counts': row_counts,
            'column_counts': column_counts,
            'sample_columns': sample_columns,
            'file_types': sorted(file_types) if file_types else [],
            'has_instance_values': has_instance_values,
            'non_empty_ratio': non_empty_ratio,
            'avg_text_length': avg_text_length,
            'noisy_text_ratio': noisy_text_ratio,
            'dirty_pattern_ratio': dirty_pattern_ratio,
            'metadata_like_schema': metadata_like_schema,
            'dataset_notes': ', '.join(notes) if notes else 'content-inspected dataset',
            'locator_hint': located['locator'],
        }

    def _save_model_artifact(self, dataset_name: str, artifact: Dict) -> Path:
        artifact_path = self._model_artifact_path(dataset_name)
        with open(artifact_path, 'wb') as f:
            pickle.dump(artifact, f)
        logger.info(f"模型已保存: {artifact_path}")
        return artifact_path

    def load_model_artifact(self, dataset_name: str) -> Optional[Dict]:
        artifact_path = self._model_artifact_path(dataset_name)
        if not artifact_path.exists():
            return None

        with open(artifact_path, 'rb') as f:
            return pickle.load(f)

    def _extract_model_scores(self, model, X: np.ndarray) -> np.ndarray:
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, 'decision_function'):
            raw_scores = model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-raw_scores))
        return model.predict(X).astype(float)

    def _should_use_smd_llm_rerank(self, candidate_rows: List[Dict]) -> bool:
        if not candidate_rows:
            return False
        scores = sorted(
            (float(row.get("ranking_score", row.get("confidence", 0.0))) for row in candidate_rows),
            reverse=True,
        )
        top_score = scores[0]
        if top_score >= 0.86:
            return False
        score_gap = (scores[0] - scores[1]) if len(scores) >= 2 else 1.0
        if len(scores) >= 2 and score_gap <= 0.06:
            return True
        if self._is_generic_target_field(
            candidate_rows[0].get("target_table", ""),
            candidate_rows[0].get("target_column", ""),
        ) and top_score <= 0.76:
            return True
        return 0.42 <= top_score <= 0.72

    def _compute_smd_ranking_score(self, candidate_row: Dict) -> float:
        source_col = self._normalize_text(candidate_row.get("source_column", ""))
        target_col = self._normalize_text(candidate_row.get("target_column", ""))
        source_table = self._normalize_text(candidate_row.get("source_table", ""))
        target_table = self._normalize_text(candidate_row.get("target_table", ""))
        source_desc = self._normalize_text(candidate_row.get("source_desc", ""))
        target_desc = self._normalize_text(candidate_row.get("target_desc", ""))
        source_type = self._normalize_text(candidate_row.get("source_type", ""))
        target_type = self._normalize_text(candidate_row.get("target_type", ""))

        name_sim = self.similarity_metrics.string_similarity(source_col, target_col)
        token_sim = self.similarity_metrics.token_overlap(source_col, target_col)
        table_sim = self.similarity_metrics.string_similarity(source_table, target_table)
        desc_sim = self.similarity_metrics.string_similarity(source_desc, target_desc)
        desc_token = self.similarity_metrics.token_overlap(source_desc, target_desc)
        semantic_sim = self._schema_semantic_similarity(
            source_table,
            source_col,
            source_desc,
            target_table,
            target_col,
            target_desc,
        )
        embedding_sim = self._schema_embedding_similarity(
            source_table,
            source_col,
            source_desc,
            source_type,
            target_table,
            target_col,
            target_desc,
            target_type,
        )
        source_role = self._infer_schema_role(source_table, source_col, source_desc)
        target_role = self._infer_schema_role(target_table, target_col, target_desc)
        role_sim = self._schema_role_similarity(source_role, target_role)
        source_domain = self._infer_schema_domain(source_table, source_col, source_desc)
        target_domain = self._infer_schema_domain(target_table, target_col, target_desc)
        domain_sim = self._schema_domain_similarity(source_domain, target_domain)
        type_match = 1.0 if source_type and source_type == target_type else 0.0
        type_soft = self._schema_type_compatibility(source_type, target_type)
        alias_sim = self._alias_similarity(
            candidate_row.get("source_table", ""),
            candidate_row.get("source_column", ""),
            candidate_row.get("target_table", ""),
            candidate_row.get("target_column", ""),
        )
        model_score = float(candidate_row.get("confidence", 0.0))
        table_pair_prior = float(candidate_row.get("table_pair_prior", 0.0))
        column_pair_prior = float(candidate_row.get("column_pair_prior", 0.0))
        role_pair_prior = float(candidate_row.get("role_pair_prior", 0.0))
        generic_penalty = self._generic_target_penalty(
            source_table,
            source_col,
            source_desc,
            target_table,
            target_col,
            target_desc,
        )
        target_bonus = self._target_specificity_bonus(
            source_table,
            source_col,
            source_desc,
            target_table,
            target_col,
            target_desc,
        )

        ranking_score = (
            0.14 * model_score
            + 0.12 * name_sim
            + 0.05 * token_sim
            + 0.08 * table_sim
            + 0.09 * desc_sim
            + 0.04 * desc_token
            + 0.19 * semantic_sim
            + 0.16 * role_sim
            + 0.00 * type_soft
            + 0.05 * alias_sim
            + 0.06 * domain_sim
            + 0.08 * table_pair_prior
            + 0.16 * column_pair_prior
            + 0.08 * role_pair_prior
        )

        if source_type and target_type and type_soft < 0.3:
            ranking_score -= 0.04
        if type_match:
            ranking_score += 0.05
        if semantic_sim >= 0.45:
            ranking_score += 0.06
        if role_sim >= 0.9:
            ranking_score += 0.12
        elif role_sim == 0.0 and source_role != "unknown" and target_role != "unknown":
            ranking_score -= 0.06
        if source_role in {"birth_datetime", "death_datetime"} and target_role not in {source_role, "date"}:
            ranking_score -= 0.18
        if source_role == "datetime" and target_role in {"birth_datetime", "death_datetime"}:
            ranking_score -= 0.22
        if source_role == "value_number" and target_role == "value_text":
            ranking_score -= 0.14
        if source_role == "value_text" and target_role == "value_number":
            ranking_score -= 0.14
        if type_soft >= 0.6 and role_sim >= 0.55:
            ranking_score += 0.05
        if domain_sim >= 0.9:
            ranking_score += 0.07
        elif domain_sim == 0.0 and source_domain != "unknown" and target_domain != "unknown":
            ranking_score -= 0.18
        if source_domain == "time" and target_domain == "person":
            ranking_score -= 0.22
        if source_domain == "measurement" and target_domain == "person":
            ranking_score -= 0.24
        if source_domain == "drug" and target_domain in {"person", "measurement"}:
            ranking_score -= 0.16
        if table_pair_prior >= 0.30:
            ranking_score += 0.10
        elif table_pair_prior == 0.0:
            ranking_score -= 0.03
        if column_pair_prior >= 0.45:
            ranking_score += 0.14
        elif column_pair_prior >= 0.25:
            ranking_score += 0.07
        if role_pair_prior >= 0.45:
            ranking_score += 0.08
        ranking_score += target_bonus
        ranking_score -= generic_penalty
        if self._is_generic_target_field(target_table, target_col) and target_bonus <= 0.05:
            ranking_score -= 0.20
        if name_sim < 0.15 and semantic_sim < 0.15 and embedding_sim < 0.20 and desc_sim < 0.15:
            ranking_score -= 0.08

        return float(max(0.0, min(1.0, ranking_score)))

    def _select_smd_llm_candidate(
        self,
        source_row: pd.Series,
        candidate_rows: List[Dict],
    ) -> Optional[str]:
        if not candidate_rows:
            return None

        model_name = getattr(self.scene_identifier, "_ollama_model", "").strip()
        host = getattr(self.scene_identifier, "_ollama_host", "").strip().rstrip("/")
        if not model_name or not host:
            return None

        cache_key = json.dumps(
            {
                "source_table": str(source_row.get("TableName", "")).strip(),
                "source_column": str(source_row.get("ColumnName", "")).strip(),
                "candidates": [
                    {
                        "target_id": row.get("target_id", ""),
                        "score": round(float(row.get("confidence", 0.0)), 4),
                    }
                    for row in candidate_rows
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_key in self._smd_llm_rerank_cache:
            return self._smd_llm_rerank_cache[cache_key]

        source_payload = {
            "table": str(source_row.get("TableName", "")).strip(),
            "column": str(source_row.get("ColumnName", "")).strip(),
            "type": str(source_row.get("ColumnType", "")).strip(),
            "description": str(source_row.get("ColumnDesc", "")).strip(),
            "is_pk": str(source_row.get("IsPK", "")).strip(),
            "is_fk": str(source_row.get("IsFK", "")).strip(),
        }
        candidates_payload = []
        for idx, row in enumerate(candidate_rows, start=1):
            candidates_payload.append(
                {
                    "choice": idx,
                    "target_id": row["target_id"],
                    "table": row["target_table"],
                    "column": row["target_column"],
                    "type": row.get("target_type", ""),
                    "description": row.get("target_desc", ""),
                    "score": round(float(row.get("confidence", 0.0)), 4),
                }
            )

        dataset_name = str(self._current_match_context.get("dataset_name", "") if hasattr(self, "_current_match_context") else "")
        few_shot = self._get_few_shot_examples(dataset_name, source_row, k=2) if dataset_name else ""
        prompt = (
            "You are helping with schema matching between MIMIC-III and OMOP CDM in a metadata-only setting.\n"
            "Choose the single best semantic match for the source field from the candidate targets.\n"
            "Return strict JSON with keys: choice, target_id, confidence, reason.\n"
            "choice must be an integer candidate number. If none is suitable, return choice as 0.\n\n"
            f"{few_shot}\n\n" if few_shot else ""
        ) + (
            f"Source field:\n{json.dumps(source_payload, ensure_ascii=False)}\n\n"
            f"Candidate targets:\n{json.dumps(candidates_payload, ensure_ascii=False)}"
        )

        try:
            response = requests.post(
                f"{host}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                },
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            raw_text = str(payload.get("response", "")).strip()
            if not raw_text:
                return None
            cleaned_text = raw_text.strip()
            if cleaned_text.startswith("```"):
                cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text, flags=re.IGNORECASE)
                cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
                cleaned_text = cleaned_text.strip()

            parsed = None
            with suppress(Exception):
                parsed = json.loads(cleaned_text)
            if parsed is None:
                match = re.search(r'"choice"\s*:\s*(\d+)', cleaned_text)
                if match:
                    parsed = {"choice": int(match.group(1))}
                elif cleaned_text.isdigit():
                    parsed = {"choice": int(cleaned_text)}
                else:
                    target_match = re.search(r'"target_id"\s*:\s*"([^"]+)"', cleaned_text)
                    if target_match:
                        parsed = {"target_id": target_match.group(1).strip()}
            if not isinstance(parsed, dict):
                self._smd_llm_rerank_cache[cache_key] = None
                return None

            choice = int(parsed.get("choice", 0) or 0)
            if 1 <= choice <= len(candidate_rows):
                selected = str(candidate_rows[choice - 1]["target_id"]).strip()
                self._smd_llm_rerank_cache[cache_key] = selected
                return selected
            selected = str(parsed.get("target_id", "")).strip()
            valid_target_ids = {row["target_id"] for row in candidate_rows}
            if selected in valid_target_ids:
                self._smd_llm_rerank_cache[cache_key] = selected
                return selected
        except Exception:
            self._smd_llm_rerank_cache[cache_key] = None
            return None

        self._smd_llm_rerank_cache[cache_key] = None
        return None

    def _should_use_smd_llm_validation(
        self,
        source_row: pd.Series,
        candidate_rows: List[Dict],
    ) -> bool:
        if not candidate_rows:
            return False

        top1 = candidate_rows[0]
        top1_score = float(top1.get("ranking_score", top1.get("confidence", 0.0)))
        top2_score = float(candidate_rows[1].get("ranking_score", candidate_rows[1].get("confidence", 0.0))) if len(candidate_rows) > 1 else 0.0
        type_soft = self._schema_type_compatibility(
            str(source_row.get("ColumnType", "")),
            str(top1.get("target_type", "")),
        )
        role_sim = self._schema_role_similarity(
            self._infer_schema_role(
                source_row.get("TableName", ""),
                source_row.get("ColumnName", ""),
                source_row.get("ColumnDesc", ""),
            ),
            self._infer_schema_role(
                top1.get("target_table", ""),
                top1.get("target_column", ""),
                top1.get("target_desc", ""),
            ),
        )

        if top1_score < 0.68:
            return True
        if len(candidate_rows) > 1 and abs(top1_score - top2_score) <= 0.05:
            return True
        if type_soft <= 0.25:
            return True
        if role_sim == 0.0 and top1_score <= 0.74:
            return True
        if self._is_generic_target_field(top1.get("target_table", ""), top1.get("target_column", "")):
            return True
        return False

    def _should_force_smd_top2_llm_decision(
        self,
        source_row: pd.Series,
        candidate_rows: List[Dict],
    ) -> bool:
        if len(candidate_rows) < 2:
            return False

        top1 = candidate_rows[0]
        top2 = candidate_rows[1]
        top1_score = float(top1.get("ranking_score", top1.get("confidence", 0.0)))
        top2_score = float(top2.get("ranking_score", top2.get("confidence", 0.0)))
        score_gap = abs(top1_score - top2_score)
        top1_generic = self._is_generic_target_field(top1.get("target_table", ""), top1.get("target_column", ""))
        top1_type = self._schema_type_compatibility(str(source_row.get("ColumnType", "")), str(top1.get("target_type", "")))
        top2_type = self._schema_type_compatibility(str(source_row.get("ColumnType", "")), str(top2.get("target_type", "")))

        if top1_score >= 0.88:
            return False
        if score_gap <= 0.04:
            return True
        if top1_generic and score_gap <= 0.08 and top1_type <= 0.55:
            return True
        if top1_type < 0.35 and top2_type >= top1_type and score_gap <= 0.10:
            return True
        return False

    def _apply_smd_one_to_one_selection(
        self,
        rows_by_source: Dict[Tuple[str, str], List[Dict]],
        top_k_per_source: int,
    ) -> List[Dict]:
        source_order = []
        for source_key, rows in rows_by_source.items():
            if not rows:
                continue
            best_score = float(rows[0].get("ranking_score", rows[0].get("confidence", 0.0)))
            second_score = float(rows[1].get("ranking_score", rows[1].get("confidence", 0.0))) if len(rows) > 1 else 0.0
            margin = best_score - second_score
            source_order.append((best_score, margin, source_key))

        source_order.sort(key=lambda item: (item[0], item[1]), reverse=True)
        used_targets = set()
        selected_rows: List[Dict] = []

        for _, _, source_key in source_order:
            rows = rows_by_source.get(source_key, [])
            chosen = None
            for row in rows:
                if row.get("target_id") not in used_targets:
                    chosen = row
                    break
            if chosen is None and rows:
                chosen = rows[0]
            if chosen is None:
                continue
            used_targets.add(chosen.get("target_id"))
            chosen["prediction_label"] = 1
            selected_rows.append(chosen)

        selected_rows.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        return selected_rows[: max(len(selected_rows), top_k_per_source)]

    def _validate_smd_match_with_llm(
        self,
        source_row: pd.Series,
        candidate_row: Dict,
    ) -> Optional[Dict]:
        model_name = getattr(self.scene_identifier, "_ollama_model", "").strip()
        host = getattr(self.scene_identifier, "_ollama_host", "").strip().rstrip("/")
        if not model_name or not host:
            return None

        cache_key = json.dumps(
            {
                "source_table": str(source_row.get("TableName", "")).strip(),
                "source_column": str(source_row.get("ColumnName", "")).strip(),
                "target_table": str(candidate_row.get("target_table", "")).strip(),
                "target_column": str(candidate_row.get("target_column", "")).strip(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_key in self._smd_llm_validation_cache:
            return self._smd_llm_validation_cache[cache_key]

        source_payload = {
            "table": str(source_row.get("TableName", "")).strip(),
            "column": str(source_row.get("ColumnName", "")).strip(),
            "type": str(source_row.get("ColumnType", "")).strip(),
            "description": str(source_row.get("ColumnDesc", "")).strip(),
        }
        target_payload = {
            "table": str(candidate_row.get("target_table", "")).strip(),
            "column": str(candidate_row.get("target_column", "")).strip(),
            "type": str(candidate_row.get("target_type", "")).strip(),
            "description": str(candidate_row.get("target_desc", "")).strip(),
        }

        prompt = (
            "You are the final validation layer for schema matching in a metadata-only setting.\n"
            "Decide whether the target field is a semantically valid final match for the source field.\n"
            "Return strict JSON with keys: valid, confidence, reason.\n"
            "valid must be true or false.\n\n"
            f"Source field:\n{json.dumps(source_payload, ensure_ascii=False)}\n\n"
            f"Candidate target field:\n{json.dumps(target_payload, ensure_ascii=False)}"
        )
        try:
            response = requests.post(
                f"{host}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                },
                timeout=6,
            )
            response.raise_for_status()
            payload = response.json()
            raw_text = str(payload.get("response", "")).strip()
            if not raw_text:
                self._smd_llm_validation_cache[cache_key] = None
                return None
            cleaned_text = raw_text.strip()
            if cleaned_text.startswith("```"):
                cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text, flags=re.IGNORECASE)
                cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
                cleaned_text = cleaned_text.strip()

            parsed = None
            with suppress(Exception):
                parsed = json.loads(cleaned_text)
            if parsed is None:
                valid_match = re.search(r'"valid"\s*:\s*(true|false)', cleaned_text, flags=re.IGNORECASE)
                confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', cleaned_text, flags=re.IGNORECASE)
                reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', cleaned_text, flags=re.IGNORECASE)
                if valid_match:
                    parsed = {
                        "valid": valid_match.group(1).lower() == "true",
                        "confidence": float(confidence_match.group(1)) if confidence_match else 0.0,
                        "reason": reason_match.group(1).strip() if reason_match else "",
                    }
            if not isinstance(parsed, dict):
                self._smd_llm_validation_cache[cache_key] = None
                return None
            result = {
                "valid": bool(parsed.get("valid", False)),
                "confidence": float(parsed.get("confidence", 0.0)),
                "reason": str(parsed.get("reason", "")).strip(),
            }
            self._smd_llm_validation_cache[cache_key] = result
            return result
        except Exception:
            self._smd_llm_validation_cache[cache_key] = None
            return None

    def ensure_smd_split(
        self,
        dataset_name: str,
        train_ratio: float = 0.7,
        random_state: int = 42,
    ) -> Optional[Dict]:
        dataset = self.smd_loader.load_dataset(dataset_name)
        if not dataset:
            return None

        train_df = dataset.get('gold_mapping_train')
        match_df = dataset.get('gold_mapping_match')
        if train_df is not None and not train_df.empty and match_df is not None and not match_df.empty:
            return {
                'train_count': int(len(train_df)),
                'match_count': int(len(match_df)),
                'train_file': str(self._smd_dataset_path(dataset_name) / 'gold_mapping_train.csv'),
                'match_file': str(self._smd_dataset_path(dataset_name) / 'gold_mapping_match.csv'),
            }

        gold_mapping = dataset.get('gold_mapping')
        if gold_mapping is None or gold_mapping.empty:
            logger.error(f"SMD 数据集缺少 gold mapping，无法拆分: {dataset_name}")
            return None

        valid_gold = gold_mapping[
            (gold_mapping['target_table'].astype(str) != '0')
            & (gold_mapping['target_column'].astype(str) != '0')
        ].copy()
        if len(valid_gold) < 2:
            logger.error(f"SMD gold mapping 数量过少，无法拆分: {dataset_name}")
            return None

        train_df, match_df = sk_train_test_split(
            valid_gold,
            train_size=train_ratio,
            random_state=random_state,
            shuffle=True,
        )

        dataset_path = self._smd_dataset_path(dataset_name)
        train_file = dataset_path / 'gold_mapping_train.csv'
        match_file = dataset_path / 'gold_mapping_match.csv'
        train_df.to_csv(train_file, index=False, encoding='utf-8')
        match_df.to_csv(match_file, index=False, encoding='utf-8')

        logger.info(
            f"SMD 数据集已拆分: {dataset_name}, train={len(train_df)}, match={len(match_df)}"
        )
        return {
            'train_count': int(len(train_df)),
            'match_count': int(len(match_df)),
            'train_file': str(train_file),
            'match_file': str(match_file),
        }

    def _resolve_sld_tables(self, data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        file_names = list(data['files'].keys())
        dataframes = data['dataframes']

        table_a = None
        table_b = None
        for file_name, df in zip(file_names, dataframes):
            lower_name = file_name.lower()
            if 'tablea' in lower_name and table_a is None:
                table_a = df.copy()
            elif 'tableb' in lower_name and table_b is None:
                table_b = df.copy()

        if table_a is None or table_b is None:
            if len(dataframes) < 2:
                raise ValueError("SLD 数据集缺少 tableA/tableB")
            table_a = dataframes[0].copy()
            table_b = dataframes[1].copy()

        return table_a, table_b

    def _collect_labeled_pair_frames(self, data: Dict) -> List[pd.DataFrame]:
        labeled_frames = []
        for file_name, df in zip(data['files'].keys(), data['dataframes']):
            lower_name = file_name.lower()
            if ('train' in lower_name or 'valid' in lower_name or 'test' in lower_name) and 'label' in df.columns:
                labeled_frames.append(df.copy())
        return labeled_frames

    @staticmethod
    def _resolve_id_column(df: pd.DataFrame) -> Optional[str]:
        for candidate in ['id', '_id', 'ltable_id', 'rtable_id']:
            if candidate in df.columns:
                return candidate
        if len(df.columns) > 0:
            return str(df.columns[0])
        return None

    def _resolve_pair_indices(self, tableA: pd.DataFrame, tableB: pd.DataFrame, pairs_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        left_id_col = self._resolve_id_column(tableA)
        right_id_col = self._resolve_id_column(tableB)

        left_lookup = {}
        right_lookup = {}
        if left_id_col is not None:
            left_lookup = {str(value): idx for idx, value in enumerate(tableA[left_id_col].tolist())}
        if right_id_col is not None:
            right_lookup = {str(value): idx for idx, value in enumerate(tableB[right_id_col].tolist())}

        resolved = []
        kept_rows = []
        for _, row in pairs_df.iterrows():
            left_raw = row['ltable_id']
            right_raw = row['rtable_id']

            left_idx = left_lookup.get(str(left_raw))
            right_idx = right_lookup.get(str(right_raw))

            if left_idx is None:
                try:
                    candidate = int(left_raw)
                    if 0 <= candidate < len(tableA):
                        left_idx = candidate
                except Exception:
                    left_idx = None

            if right_idx is None:
                try:
                    candidate = int(right_raw)
                    if 0 <= candidate < len(tableB):
                        right_idx = candidate
                except Exception:
                    right_idx = None

            if left_idx is None or right_idx is None:
                continue

            resolved.append((left_idx, right_idx))
            kept_rows.append(row.to_dict())

        resolved_array = np.array(resolved, dtype=int) if resolved else np.empty((0, 2), dtype=int)
        kept_df = pd.DataFrame(kept_rows) if kept_rows else pairs_df.iloc[0:0].copy()
        return resolved_array, kept_df

    def _prepare_sld_prediction_data(
        self,
        category: str,
        dataset_name: str,
        normalize_before_match: bool = False,
    ) -> Optional[Dict]:
        data = self.dataset_loader.load_dataset(category, dataset_name)
        if not data:
            return None

        tableA, tableB = self._resolve_sld_tables(data)
        if normalize_before_match:
            tableA = DataPreprocessor.normalize_data(tableA)
            tableB = DataPreprocessor.normalize_data(tableB)
        labeled_frames = self._collect_labeled_pair_frames(data)
        if not labeled_frames:
            logger.error(f"SLD 数据集未找到带 label 的候选配对文件: {dataset_name}")
            return None

        pairs_df = pd.concat(labeled_frames, ignore_index=True)
        if normalize_before_match:
            pairs_df = DataPreprocessor.normalize_data(pairs_df)
        if 'ltable_id' not in pairs_df.columns or 'rtable_id' not in pairs_df.columns:
            logger.error("SLD 配对文件缺少 ltable_id / rtable_id 列")
            return None

        pair_indices, pairs_df = self._resolve_pair_indices(tableA, tableB, pairs_df)
        if len(pair_indices) == 0:
            logger.error("SLD 配对文件中的 ID 无法映射到表记录")
            return None

        X = self.extract_advanced_features(tableA, tableB, pair_indices)
        if len(X) == 0:
            logger.error("SLD 预测特征提取失败")
            return None

        return {
            'tableA': tableA,
            'tableB': tableB,
            'pairs_df': pairs_df,
            'X': X,
        }

    def predict_sld_matches(
        self,
        dataset_name: str,
        category: str,
        threshold: float = 0.5,
        normalize_before_match: bool = False,
    ) -> Optional[Dict]:
        artifact = self.load_model_artifact(dataset_name)
        if artifact is None:
            logger.info(f"未找到 {dataset_name} 的已训练模型，先自动训练")
            train_result = self.train_model(category, dataset_name)
            if train_result is None:
                return None
            artifact = self.load_model_artifact(dataset_name)
            if artifact is None:
                logger.error("模型训练成功但未找到持久化产物")
                return None

        prepared = self._prepare_sld_prediction_data(
            category,
            dataset_name,
            normalize_before_match=normalize_before_match,
        )
        if prepared is None:
            return None

        model = artifact['model']
        pairs_df = prepared['pairs_df'].copy()
        scores = self._extract_model_scores(model, prepared['X'])
        predictions = (scores >= threshold).astype(int)

        pairs_df['prediction_score'] = scores
        pairs_df['prediction_label'] = predictions
        positive_df = pairs_df[pairs_df['prediction_label'] == 1].copy()
        positive_df = positive_df.sort_values('prediction_score', ascending=False)

        matches = []
        for _, row in positive_df.iterrows():
            match = {
                'ltable_id': int(row['ltable_id']),
                'rtable_id': int(row['rtable_id']),
                'confidence': float(row['prediction_score']),
                'match_method': f"sld_model_prediction:{artifact['model_type']}",
            }

            left_columns = [col for col in positive_df.columns if col.startswith('ltable_')]
            right_columns = [col for col in positive_df.columns if col.startswith('rtable_')]
            for col in left_columns[:6]:
                match[col] = row[col]
            for col in right_columns[:6]:
                match[col] = row[col]

            if 'label' in row.index:
                match['gold_label'] = int(row['label'])

            matches.append(match)

        metrics = {}
        if 'label' in pairs_df.columns:
            y_true = pairs_df['label'].astype(int).values
            y_pred = pairs_df['prediction_label'].astype(int).values
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'evaluated_pair_count': int(len(pairs_df)),
            }

        return {
            'dataset': dataset_name,
            'category': category,
            'data_type': 'SLD',
            'match_count': len(matches),
            'threshold': threshold,
            'normalize_before_match': bool(normalize_before_match),
            'model_type': artifact['model_type'],
            'matches': matches,
            'evaluation': metrics,
        }

    def predict_smd_matches(
        self,
        dataset_name: str,
        threshold: float = 0.5,
        top_k_per_source: int = 3,
        split_role: str = 'match',
        enable_llm_rerank: bool = True,
    ) -> Optional[Dict]:
        artifact = self.load_model_artifact(dataset_name)
        if artifact is None:
            logger.info(f"未找到 {dataset_name} 的已训练模型，先自动训练")
            train_result = self.train_smd_model(dataset_name)
            if train_result is None:
                return None
            artifact = self.load_model_artifact(dataset_name)
            if artifact is None:
                logger.error("模型训练成功但未找到持久化产物")
                return None

        dataset = self.smd_loader.load_dataset(dataset_name)
        if not dataset or len(dataset['sources']) < 2:
            logger.error(f"无法加载 SMD 数据集用于预测: {dataset_name}")
            return None

        split_info = self.ensure_smd_split(dataset_name)
        dataset = self.smd_loader.load_dataset(dataset_name)
        holdout_mapping = dataset.get('gold_mapping_match') if split_role != 'train' else dataset.get('gold_mapping_train')
        if holdout_mapping is None or holdout_mapping.empty:
            holdout_mapping = dataset.get('gold_mapping_match') or dataset.get('gold_mapping_train')
        if holdout_mapping is None or holdout_mapping.empty:
            logger.error(f"SMD 数据集缺少匹配保留集: {dataset_name}")
            return None

        source_names = list(dataset['sources'].keys())
        source_name = artifact.get('source_name', source_names[0])
        target_name = artifact.get('target_name', source_names[1])
        source_df = dataset['sources'][source_name]['data']
        target_df = dataset['sources'][target_name]['data']
        holdout_mapping = holdout_mapping[
            (holdout_mapping['target_table'].astype(str) != '0')
            & (holdout_mapping['target_column'].astype(str) != '0')
        ].copy()
        if holdout_mapping.empty:
            logger.error("SMD 保留集为空")
            return None

        holdout_source_keys = {
            (str(row['source_table']), str(row['source_column']))
            for _, row in holdout_mapping.iterrows()
        }
        gold_lookup = {}
        prior_maps = self._build_smd_prior_maps(dataset_name, dataset, source_name, target_name)
        table_pair_priors = prior_maps.get('table_pair_priors', {}) or artifact.get('table_pair_priors', {}) or {}
        column_pair_priors = prior_maps.get('column_pair_priors', {})
        role_pair_priors = prior_maps.get('role_pair_priors', {})
        for _, row in holdout_mapping.iterrows():
            source_key = (str(row['source_table']), str(row['source_column']))
            gold_lookup.setdefault(source_key, set()).add(
                (str(row['target_table']), str(row['target_column']))
            )

        # 先用轻量级名称相似度做候选裁剪，避免对全部字段笛卡尔积都跑模型。
        candidate_limit = min(
            max(top_k_per_source * 100, 200),
            max(len(target_df), top_k_per_source),
        )
        feature_rows = []
        pair_rows = []
        for _, src_row in source_df.iterrows():
            source_key = (str(src_row.get('TableName', '')), str(src_row.get('ColumnName', '')))
            if source_key not in holdout_source_keys:
                continue

            shortlisted_targets = []
            for _, tgt_row in target_df.iterrows():
                col_name_sim = self.similarity_metrics.string_similarity(
                    self._normalize_text(src_row.get('ColumnName', '')),
                    self._normalize_text(tgt_row.get('ColumnName', '')),
                )
                col_token_sim = self.similarity_metrics.token_overlap(
                    self._normalize_text(src_row.get('ColumnName', '')),
                    self._normalize_text(tgt_row.get('ColumnName', '')),
                )
                table_name_sim = self.similarity_metrics.string_similarity(
                    self._normalize_text(src_row.get('TableName', '')),
                    self._normalize_text(tgt_row.get('TableName', '')),
                )
                desc_name_sim = self.similarity_metrics.string_similarity(
                    self._normalize_text(src_row.get('ColumnDesc', '')),
                    self._normalize_text(tgt_row.get('ColumnDesc', '')),
                )
                semantic_sim = self._schema_semantic_similarity(
                    src_row.get('TableName', ''),
                    src_row.get('ColumnName', ''),
                    src_row.get('ColumnDesc', ''),
                    tgt_row.get('TableName', ''),
                    tgt_row.get('ColumnName', ''),
                    tgt_row.get('ColumnDesc', ''),
                )
                embedding_sim = self._schema_embedding_similarity(
                    src_row.get('TableName', ''),
                    src_row.get('ColumnName', ''),
                    src_row.get('ColumnDesc', ''),
                    src_row.get('ColumnType', ''),
                    tgt_row.get('TableName', ''),
                    tgt_row.get('ColumnName', ''),
                    tgt_row.get('ColumnDesc', ''),
                    tgt_row.get('ColumnType', ''),
                )
                role_sim = self._schema_role_similarity(
                    self._infer_schema_role(
                        src_row.get('TableName', ''),
                        src_row.get('ColumnName', ''),
                        src_row.get('ColumnDesc', ''),
                    ),
                    self._infer_schema_role(
                        tgt_row.get('TableName', ''),
                        tgt_row.get('ColumnName', ''),
                        tgt_row.get('ColumnDesc', ''),
                    ),
                )
                source_domain = self._infer_schema_domain(
                    src_row.get('TableName', ''),
                    src_row.get('ColumnName', ''),
                    src_row.get('ColumnDesc', ''),
                )
                target_domain = self._infer_schema_domain(
                    tgt_row.get('TableName', ''),
                    tgt_row.get('ColumnName', ''),
                    tgt_row.get('ColumnDesc', ''),
                )
                source_role = self._infer_schema_role(
                    src_row.get('TableName', ''),
                    src_row.get('ColumnName', ''),
                    src_row.get('ColumnDesc', ''),
                )
                target_role = self._infer_schema_role(
                    tgt_row.get('TableName', ''),
                    tgt_row.get('ColumnName', ''),
                    tgt_row.get('ColumnDesc', ''),
                )
                domain_sim = self._schema_domain_similarity(source_domain, target_domain)
                source_type = self._normalize_text(src_row.get('ColumnType', ''))
                target_type = self._normalize_text(tgt_row.get('ColumnType', ''))
                type_soft = self._schema_type_compatibility(source_type, target_type)
                source_table_norm = self._normalize_text(src_row.get('TableName', ''))
                target_table_norm = self._normalize_text(tgt_row.get('TableName', ''))
                source_col_norm = self._normalize_text(src_row.get('ColumnName', ''))
                target_col_norm = self._normalize_text(tgt_row.get('ColumnName', ''))
                table_prior_live = float(table_pair_priors.get(f"{source_table_norm}::{target_table_norm}", 0.0))
                column_prior_live = float(column_pair_priors.get(f"{source_col_norm}::{target_col_norm}", 0.0))
                role_prior_live = float(role_pair_priors.get(f"{source_role}::{target_role}", 0.0))
                alias_sim = self._alias_similarity(
                    src_row.get('TableName', ''),
                    src_row.get('ColumnName', ''),
                    tgt_row.get('TableName', ''),
                    tgt_row.get('ColumnName', ''),
                )
                generic_penalty = self._generic_target_penalty(
                    self._normalize_text(src_row.get('TableName', '')),
                    self._normalize_text(src_row.get('ColumnName', '')),
                    self._normalize_text(src_row.get('ColumnDesc', '')),
                    self._normalize_text(tgt_row.get('TableName', '')),
                    self._normalize_text(tgt_row.get('ColumnName', '')),
                    self._normalize_text(tgt_row.get('ColumnDesc', '')),
                )
                pre_score = (
                    0.08 * col_name_sim
                    + 0.05 * col_token_sim
                    + 0.06 * table_name_sim
                    + 0.12 * desc_name_sim
                    + 0.24 * semantic_sim
                    + 0.18 * role_sim
                    + 0.06 * type_soft
                    + 0.07 * alias_sim
                    + 0.12 * table_prior_live
                    + 0.18 * column_prior_live
                    + 0.12 * role_prior_live
                )
                if role_sim == 0.0 and semantic_sim < 0.1 and col_name_sim < 0.1 and desc_name_sim < 0.1:
                    pre_score -= 0.04
                shortlisted_targets.append((pre_score, tgt_row))

            shortlisted_targets.sort(key=lambda item: item[0], reverse=True)
            for _, tgt_row in shortlisted_targets[:candidate_limit]:
                feature_rows.append(self._extract_smd_field_pair_features(src_row, tgt_row))
                pair_rows.append((src_row, tgt_row))

        if not feature_rows:
            logger.error("SMD 候选字段配对为空")
            return None

        X = np.vstack(feature_rows)
        model_feature_count = len(artifact.get('feature_names', []))
        if model_feature_count and X.shape[1] != model_feature_count:
            logger.info(
                f"{dataset_name} SMD feature mismatch: artifact={model_feature_count}, current={X.shape[1]}; retraining"
            )
            train_result = self.train_smd_model(dataset_name, artifact.get('model_type', 'gradient_boosting'))
            if train_result is None:
                return None
            return self.predict_smd_matches(
                dataset_name,
                threshold=threshold,
                top_k_per_source=top_k_per_source,
                split_role=split_role,
                enable_llm_rerank=enable_llm_rerank,
            )
        scaler = artifact.get('scaler')
        if scaler is not None:
            X = scaler.transform(X)

        scores = self._extract_model_scores(artifact['model'], X)
        candidate_rows = []
        for score, (src_row, tgt_row) in zip(scores, pair_rows):
            source_key = (str(src_row.get('TableName', '')), str(src_row.get('ColumnName', '')))
            target_key = (str(tgt_row.get('TableName', '')), str(tgt_row.get('ColumnName', '')))
            source_role = self._infer_schema_role(
                src_row.get('TableName', ''),
                src_row.get('ColumnName', ''),
                src_row.get('ColumnDesc', ''),
            )
            target_role = self._infer_schema_role(
                tgt_row.get('TableName', ''),
                tgt_row.get('ColumnName', ''),
                tgt_row.get('ColumnDesc', ''),
            )
            source_table_norm = self._normalize_text(source_key[0])
            target_table_norm = self._normalize_text(target_key[0])
            source_col_norm = self._normalize_text(source_key[1])
            target_col_norm = self._normalize_text(target_key[1])
            is_gold = int(target_key in gold_lookup.get(source_key, set()))
            candidate_rows.append({
                'source_table': source_key[0],
                'source_column': source_key[1],
                'source_id': f"{source_key[0]}::{source_key[1]}",
                'target_table': target_key[0],
                'target_column': target_key[1],
                'target_id': f"{target_key[0]}::{target_key[1]}",
                'source_desc': str(src_row.get('ColumnDesc', '')),
                'target_desc': str(tgt_row.get('ColumnDesc', '')),
                'source_type': str(src_row.get('ColumnType', '')),
                'target_type': str(tgt_row.get('ColumnType', '')),
                'table_pair_prior': float(table_pair_priors.get(f"{source_table_norm}::{target_table_norm}", 0.0)),
                'column_pair_prior': float(column_pair_priors.get(f"{source_col_norm}::{target_col_norm}", 0.0)),
                'role_pair_prior': float(role_pair_priors.get(f"{source_role}::{target_role}", 0.0)),
                'confidence': float(score),
                'match_method': f"smd_model_prediction:{artifact['model_type']}",
                'gold_label': is_gold,
            })

        for row in candidate_rows:
            row['ranking_score'] = self._compute_smd_ranking_score(row)

        by_source = {}
        for row in candidate_rows:
            source_key = (row['source_table'], row['source_column'])
            by_source.setdefault(source_key, []).append(row)

        matches = []
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        reciprocal_rank_sum = 0.0
        llm_reranked_sources = 0
        llm_validated_sources = 0
        llm_call_budget = 6 if enable_llm_rerank else 0
        llm_call_count = 0
        llm_budget_notice_shown = False
        finalized_rows_by_source: Dict[Tuple[str, str], List[Dict]] = {}
        total_sources = len(by_source)
        if enable_llm_rerank:
            print(f"[SMD] LLM 精判预算: 最多 {llm_call_budget} 次调用")
        for source_index, (source_key, rows) in enumerate(by_source.items(), start=1):
            if source_index == 1 or source_index % 25 == 0 or source_index == total_sources:
                print(f"[SMD] 基础匹配进度: {source_index}/{total_sources}")
            rows.sort(key=lambda item: item.get('ranking_score', item['confidence']), reverse=True)
            source_lookup_row = source_df[
                (source_df['TableName'].astype(str) == source_key[0])
                & (source_df['ColumnName'].astype(str) == source_key[1])
            ]
            llm_used_for_source = False
            if (
                enable_llm_rerank
                and llm_call_count < llm_call_budget
                and not llm_used_for_source
                and not source_lookup_row.empty
                and self._should_force_smd_top2_llm_decision(source_lookup_row.iloc[0], rows[: min(2, len(rows))])
            ):
                selected_target_id = self._select_smd_llm_candidate(source_lookup_row.iloc[0], rows[: min(2, len(rows))])
                llm_call_count += 1
                llm_used_for_source = True
                if selected_target_id:
                    llm_reranked_sources += 1
                    for row in rows:
                        if row['target_id'] == selected_target_id:
                            row['confidence'] = min(1.0, float(row['confidence']) + 0.18)
                            row['ranking_score'] = min(1.0, float(row.get('ranking_score', row['confidence'])) + 0.24)
                            row['match_method'] = f"{row['match_method']}+llm_top2"
                            row['llm_selected'] = True
                        elif row.get('llm_selected') is not True and row['target_id'] in {item['target_id'] for item in rows[:2]}:
                            row['ranking_score'] = max(0.0, float(row.get('ranking_score', row['confidence'])) - 0.08)
                    rows.sort(key=lambda item: item.get('ranking_score', item['confidence']), reverse=True)
            if enable_llm_rerank:
                llm_shortlist = rows[: min(3, len(rows))]
                if (
                    llm_call_count < llm_call_budget
                    and not llm_used_for_source
                    and
                    not source_lookup_row.empty
                    and llm_shortlist
                    and self._should_use_smd_llm_rerank(llm_shortlist)
                ):
                    selected_target_id = self._select_smd_llm_candidate(source_lookup_row.iloc[0], llm_shortlist)
                    llm_call_count += 1
                    llm_used_for_source = True
                    if selected_target_id:
                        llm_reranked_sources += 1
                        for row in rows:
                            if row['target_id'] == selected_target_id:
                                row['confidence'] = min(1.0, float(row['confidence']) + 0.12)
                                row['ranking_score'] = min(1.0, float(row.get('ranking_score', row['confidence'])) + 0.2)
                                row['match_method'] = f"{row['match_method']}+llm_rerank"
                                row['llm_selected'] = True
                                break
                        rows.sort(key=lambda item: item.get('ranking_score', item['confidence']), reverse=True)

            if (
                enable_llm_rerank
                and llm_call_count < llm_call_budget
                and not llm_used_for_source
                and rows
                and not source_lookup_row.empty
                and self._should_use_smd_llm_validation(source_lookup_row.iloc[0], rows[: min(3, len(rows))])
            ):
                validation = self._validate_smd_match_with_llm(source_lookup_row.iloc[0], rows[0])
                llm_call_count += 1
                llm_used_for_source = True
                if validation is not None:
                    llm_validated_sources += 1
                    rows[0]['llm_validation'] = validation
                    rows[0]['match_method'] = f"{rows[0]['match_method']}+llm_validate"
                    if validation.get('valid'):
                        rows[0]['confidence'] = min(1.0, float(rows[0]['confidence']) + 0.08)
                        rows[0]['ranking_score'] = min(
                            1.0,
                            float(rows[0].get('ranking_score', rows[0]['confidence'])) + 0.08,
                        )
                        rows[0]['llm_validated'] = True
                    else:
                        rows[0]['confidence'] = max(0.0, float(rows[0]['confidence']) - 0.12)
                        rows[0]['ranking_score'] = max(
                            0.0,
                            float(rows[0].get('ranking_score', rows[0]['confidence'])) - 0.18,
                        )
                        rows[0]['llm_validated'] = False
                    rows.sort(key=lambda item: item.get('ranking_score', item['confidence']), reverse=True)
            if enable_llm_rerank and llm_call_count >= llm_call_budget and not llm_budget_notice_shown:
                print("[SMD] LLM 精判预算已用尽，后续源字段仅使用快速排序。")
                llm_budget_notice_shown = True

            effective_threshold = threshold
            if source_key in holdout_source_keys:
                effective_threshold = max(0.34, threshold - 0.06)
            kept = [row for row in rows if row.get('ranking_score', row['confidence']) >= effective_threshold][:top_k_per_source]
            if not kept and rows:
                kept = rows[:top_k_per_source]
            expected_targets = sorted(gold_lookup.get(source_key, set()))
            ranked_target_ids = [(row['target_table'], row['target_column']) for row in rows]
            first_hit_rank = 0
            for rank_idx, target in enumerate(ranked_target_ids, start=1):
                if target in expected_targets:
                    first_hit_rank = rank_idx
                    break
            if any(target in expected_targets for target in ranked_target_ids[:1]):
                top1_correct += 1
            if any(target in expected_targets for target in ranked_target_ids[:3]):
                top3_correct += 1
            if any(target in expected_targets for target in ranked_target_ids[:5]):
                top5_correct += 1
                if first_hit_rank:
                    reciprocal_rank_sum += 1.0 / first_hit_rank
            for row in rows:
                row['source_id'] = f"{row['source_table']}::{row['source_column']}"
                row['target_id'] = f"{row['target_table']}::{row['target_column']}"
                row['expected_targets'] = [
                    f"{table}::{column}" for table, column in expected_targets
                ]
            finalized_rows_by_source[source_key] = rows

        matches = self._apply_smd_one_to_one_selection(finalized_rows_by_source, top_k_per_source)
        evaluated_source_count = len(holdout_source_keys)
        predicted_positive = len(matches)
        true_positive = sum(int(row['gold_label']) for row in matches)
        candidate_precision = (true_positive / predicted_positive) if predicted_positive else 0.0
        source_level_recall = (true_positive / evaluated_source_count) if evaluated_source_count else 0.0
        if candidate_precision + source_level_recall > 0:
            source_level_f1 = (
                2 * candidate_precision * source_level_recall
                / (candidate_precision + source_level_recall)
            )
        else:
            source_level_f1 = 0.0
        top1_accuracy = (top1_correct / evaluated_source_count) if evaluated_source_count else 0.0
        top3_accuracy = (top3_correct / evaluated_source_count) if evaluated_source_count else 0.0
        top5_accuracy = (top5_correct / evaluated_source_count) if evaluated_source_count else 0.0
        mrr = (reciprocal_rank_sum / evaluated_source_count) if evaluated_source_count else 0.0

        return {
            'dataset': dataset_name,
            'data_type': 'SMD',
            'match_count': len(matches),
            'threshold': threshold,
            'top_k_per_source': top_k_per_source,
            'candidate_limit_per_source': candidate_limit,
            'model_type': artifact['model_type'],
            'split_role': split_role,
            'enable_llm_rerank': bool(enable_llm_rerank),
            'llm_reranked_source_count': int(llm_reranked_sources),
            'llm_validated_source_count': int(llm_validated_sources),
            'split_summary': split_info or {},
            'holdout_pair_count': int(len(holdout_mapping)),
            'evaluated_source_count': int(evaluated_source_count),
            'matches': matches,
            'evaluation': {
                'metric_family': 'ranking',
                'accuracy': float(top1_accuracy),
                'precision': float(candidate_precision),
                'recall': float(source_level_recall),
                'f1_score': float(source_level_f1),
                'top1_accuracy': float(top1_accuracy),
                'top3_accuracy': float(top3_accuracy),
                'top5_accuracy': float(top5_accuracy),
                'hit_at_1': float(top1_accuracy),
                'hit_at_3': float(top3_accuracy),
                'hit_at_5': float(top5_accuracy),
                'mrr': float(mrr),
                'candidate_precision': float(candidate_precision),
                'predicted_positive_count': int(predicted_positive),
                'true_positive_count': int(true_positive),
            },
        }

    def match_single_dataset(
        self,
        dataset_name: str,
        threshold: float = 0.5,
        split_role: str = 'match',
        normalize_before_match: bool = False,
        top_k_per_source: int = 3,
        enable_llm_rerank: bool = True,
    ) -> Optional[Dict]:
        self._current_match_context = {
            "dataset_name": dataset_name,
            "split_role": split_role,
        }
        scene_info = self.identify_dataset_scene(dataset_name)
        if scene_info is None:
            logger.error(f"数据集不存在: {dataset_name}")
            return None

        if scene_info['type'] == 'SMD':
            result = self.predict_smd_matches(
                dataset_name,
                threshold=threshold,
                split_role=split_role,
                top_k_per_source=top_k_per_source,
                enable_llm_rerank=enable_llm_rerank,
            )
        else:
            result = self.predict_sld_matches(
                dataset_name,
                scene_info['category'],
                threshold=threshold,
                normalize_before_match=normalize_before_match,
            )

        if result is None:
            return None

        result['scene'] = scene_info['scene']
        result['matching_strategy'] = scene_info['matching_strategy']
        result['scene_detector'] = scene_info['scene_detector']
        result['scene_confidence'] = scene_info['scene_confidence']
        result['scene_evidence'] = scene_info['scene_evidence']
        return result

    def identify_dataset_scene(self, dataset_name: str) -> Optional[Dict]:
        """根据数据集内容自动识别场景并给出建议策略"""
        if dataset_name in self._scene_info_cache:
            return self._scene_info_cache[dataset_name].copy()

        all_datasets = self.get_all_datasets_info()
        if dataset_name not in all_datasets:
            return None

        dataset_info = all_datasets[dataset_name].copy()
        storage_category = dataset_info.get('category', '')
        dataset_payload = self._build_scene_payload(dataset_name, dataset_info)
        decision = self.scene_identifier.identify(dataset_name, dataset_payload)

        dataset_info.update({
            'type': decision.data_type,
            'scene': decision.scene,
            'description': (
                '仅元数据场景，使用 schema/字段级特征进行匹配'
                if decision.data_type == 'SMD'
                else '带实例数据场景，使用实例值相似度与上下文特征进行匹配'
            ),
            'matching_strategy': decision.matching_strategy,
            'recommended_model': decision.recommended_model,
            'scene_detector': decision.detector,
            'scene_confidence': decision.confidence,
            'scene_evidence': decision.evidence,
            'storage_category': storage_category,
            'quality_label': decision.quality_label,
            'quality_evidence': decision.quality_evidence,
            'data_quality_risk': decision.quality_label == 'dirty_like',
        })
        self._scene_info_cache[dataset_name] = dataset_info.copy()
        return dataset_info

    def _build_scene_payload(self, dataset_name: str, dataset_info: Dict) -> Dict:
        payload = self._build_content_only_scene_payload(dataset_name)
        if payload:
            return payload
        return {}
    
    def extract_advanced_features(self, tableA: pd.DataFrame, tableB: pd.DataFrame, 
                                 pair_indices: np.ndarray) -> np.ndarray:
        """
        提取高级特征 - 融合多种相似度度量
        
        特征包括：
        1. 字符串相似度（SequenceMatcher）
        2. N-gram Jaccard 相似度
        3. 编辑距离
        4. 词汇重叠
        5. 上下文相似度
        6. 音韵相似度
        7. 数据类型兼容性
        """
        features = []
        
        common_columns = set(tableA.columns) & set(tableB.columns)
        # 过滤掉ID列
        common_columns = {c for c in common_columns 
                         if c.lower() not in ['_id', 'id', 'index', 'label', 'match']}
        
        if not common_columns:
            # 如果没有共同列，返回随机特征（作为备选）
            logger.warning("未找到共同的非ID列")
            common_columns = set(tableA.columns) & set(tableB.columns)
        
        for idx_a, idx_b in pair_indices:
            feature_vector = []
            
            try:
                for col in sorted(common_columns):
                    try:
                        val_a = str(tableA.iloc[idx_a][col]).strip()
                        val_b = str(tableB.iloc[idx_b][col]).strip()
                        
                        # 处理缺失值
                        if val_a.lower() == 'nan' or val_a == '':
                            val_a = ''
                        if val_b.lower() == 'nan' or val_b == '':
                            val_b = ''
                        
                        # 1. 精确匹配
                        exact_match = 1.0 if val_a == val_b else 0.0
                        feature_vector.append(exact_match)
                        
                        # 2. 字符串相似度 (SequenceMatcher)
                        seq_sim = self.similarity_metrics.string_similarity(val_a, val_b)
                        feature_vector.append(seq_sim)
                        
                        # 3. Jaccard 相似度 (2-gram)
                        jaccard_sim = self.similarity_metrics.jaccard_similarity(val_a, val_b, 2)
                        feature_vector.append(jaccard_sim)
                        
                        # 4. Jaccard 相似度 (3-gram)
                        jaccard_3 = self.similarity_metrics.jaccard_similarity(val_a, val_b, 3)
                        feature_vector.append(jaccard_3)
                        
                        # 5. 编辑距离
                        lev_dist = self.similarity_metrics.levenshtein_distance(val_a, val_b)
                        feature_vector.append(lev_dist)
                        
                        # 6. 词汇重叠
                        token_sim = self.similarity_metrics.token_overlap(val_a, val_b)
                        feature_vector.append(token_sim)
                        
                        # 7. 上下文相似度
                        context_sim = self.similarity_metrics.contextual_similarity(val_a, val_b)
                        feature_vector.append(context_sim)
                        
                        # 8. 音韵相似度
                        phonetic_sim = self.similarity_metrics.phonetic_similarity(val_a, val_b)
                        feature_vector.append(phonetic_sim)
                        
                        # 9. 数据类型兼容性
                        type_compat = self.similarity_metrics.data_type_compatibility(val_a, val_b)
                        feature_vector.append(type_compat)
                        
                        # 10. 长度比较
                        if max(len(val_a), len(val_b)) > 0:
                            len_ratio = min(len(val_a), len(val_b)) / max(len(val_a), len(val_b))
                        else:
                            len_ratio = 1.0 if val_a == val_b else 0.0
                        feature_vector.append(len_ratio)
                        
                    except Exception as e:
                        # 如果单个列处理失败，添加默认特征
                        feature_vector.extend([0.0] * 10)
                        logger.debug(f"列处理失败 {col}: {e}")
                
                if feature_vector:
                    features.append(feature_vector)
                else:
                    features.append([0.0] * 10)
                    
            except Exception as e:
                logger.debug(f"行处理失败: {e}")
                features.append([0.0] * 10)
        
        return np.array(features) if features else np.array([]).reshape(0, 10)
    
    def detect_data_type(self, tableA: pd.DataFrame, tableB: pd.DataFrame) -> str:
        """检测数据集类型
        
        SLD (Schema with instance Data)：有实例数据
        SMD (Schema with only MetaData)：仅有字段名，无实例数据
        """
        # 检查是否有足够的数据值（非null/nan值）
        common_columns = set(tableA.columns) & set(tableB.columns)
        common_columns = {c for c in common_columns 
                         if c.lower() not in ['_id', 'id', 'index', 'label', 'match']}
        
        if not common_columns:
            return "SMD"
        
        # 计算非空值的比例
        non_empty_count = 0
        total_cells = 0
        
        for col in common_columns:
            if col in tableA.columns:
                non_empty_A = tableA[col].notna().sum()
                non_empty_count += non_empty_A
                total_cells += len(tableA)
            if col in tableB.columns:
                non_empty_B = tableB[col].notna().sum()
                non_empty_count += non_empty_B
                total_cells += len(tableB)
        
        # 如果有超过30%的非空数据值，判定为SLD
        data_ratio = non_empty_count / total_cells if total_cells > 0 else 0
        return "SLD" if data_ratio > 0.3 else "SMD"

    @staticmethod
    def _normalize_text(value: object) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip().lower()
        text = re.sub(r"[_\-/]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _schema_text_for_embedding(
        self,
        table_name: object,
        column_name: object,
        description: object,
        column_type: object = "",
    ) -> str:
        parts = [
            self._normalize_text(table_name),
            self._normalize_text(column_name),
            self._normalize_text(description),
            self._normalize_text(column_type),
        ]
        return " ".join(part for part in parts if part).strip()

    @staticmethod
    def _tokenize_schema_text(*values: object) -> List[str]:
        tokens: List[str] = []
        for value in values:
            text = ImprovedSchemaMatchingTrainer._normalize_text(value)
            if not text:
                continue
            parts = re.findall(r"[a-z0-9]+", text)
            tokens.extend(parts)
        return [token for token in tokens if token]

    def _expand_schema_tokens(self, tokens: List[str]) -> set:
        synonym_map = {
            "subject": {"person", "patient"},
            "patient": {"person", "subject"},
            "person": {"patient", "subject"},
            "hadm": {"visit", "occurrence", "admission"},
            "admission": {"visit", "occurrence", "hadm"},
            "visit": {"occurrence", "admission", "hadm"},
            "icustay": {"visit", "detail", "stay"},
            "stay": {"visit", "detail"},
            "itemid": {"concept", "code", "source", "value"},
            "cgid": {"provider", "caregiver"},
            "caregiver": {"provider", "cgid"},
            "dob": {"birth", "date", "datetime"},
            "dod": {"death", "date", "datetime"},
            "gender": {"sex"},
            "uom": {"unit"},
            "unit": {"uom"},
            "charttime": {"datetime", "time", "date"},
            "chartdate": {"date", "datetime"},
            "intime": {"start", "datetime", "date"},
            "outtime": {"end", "datetime", "date"},
            "starttime": {"start", "datetime", "date"},
            "endtime": {"end", "datetime", "date"},
            "value": {"result", "measurement", "observation"},
            "amount": {"dose", "quantity"},
            "provider": {"caregiver", "cgid"},
            "code": {"concept", "source"},
            "loinc": {"concept", "code"},
            "lab": {"measurement", "observation", "test"},
            "labs": {"measurement", "observation", "test"},
            "measurement": {"lab", "result", "observation", "test"},
            "observation": {"measurement", "result", "finding"},
            "result": {"value", "measurement", "observation"},
            "test": {"measurement", "lab", "observation"},
            "drug": {"medication", "prescription", "ingredient"},
            "med": {"medication", "drug", "prescription"},
            "medication": {"drug", "prescription", "ingredient"},
            "prescription": {"medication", "drug"},
            "dose": {"amount", "quantity", "strength"},
            "strength": {"dose", "amount"},
            "route": {"administration", "path"},
            "diagnosis": {"condition", "disease", "problem"},
            "diagnoses": {"condition", "disease", "problem"},
            "condition": {"diagnosis", "disease", "problem"},
            "disease": {"diagnosis", "condition"},
            "problem": {"diagnosis", "condition"},
            "procedure": {"operation", "intervention", "treatment"},
            "operation": {"procedure", "intervention"},
            "intervention": {"procedure", "operation", "treatment"},
            "treatment": {"procedure", "intervention"},
            "encounter": {"visit", "occurrence", "admission"},
            "admit": {"admission", "visit", "occurrence"},
            "discharge": {"end", "outtime"},
            "death": {"dod", "deceased"},
            "deceased": {"death", "dod"},
            "ethnicity": {"race", "demographic"},
            "race": {"ethnicity", "demographic"},
            "demographic": {"race", "ethnicity", "gender", "birth"},
            "specimen": {"sample", "material"},
            "sample": {"specimen", "material"},
            "material": {"sample", "specimen"},
            "icu": {"icustay", "stay", "detail"},
            "ward": {"location", "caresite"},
            "location": {"caresite", "ward", "site"},
            "caresite": {"location", "site", "ward"},
            "site": {"caresite", "location"},
            "providerid": {"provider", "caregiver", "cgid"},
            "patientid": {"person", "patient", "subject"},
            "subjectid": {"person", "patient", "subject"},
            "visitid": {"visit", "occurrence", "hadm"},
            "admissionid": {"visit", "occurrence", "hadm"},
            "conceptid": {"concept", "code", "identifier"},
            "sourcevalue": {"source", "verbatim", "text"},
            "valuenum": {"value", "numeric", "number", "result"},
            "valueuom": {"unit", "uom"},
            "labtime": {"charttime", "datetime", "time"},
            "storetime": {"datetime", "time"},
            "expire": {"death", "deceased", "dod", "end"},
            "admittime": {"start", "datetime", "admission", "begin"},
            "dischtime": {"end", "datetime", "discharge"},
            "icd9_code": {"concept", "code", "diagnosis", "icd"},
            "icd9": {"concept", "code", "diagnosis", "icd"},
            "drg_code": {"concept", "code", "drg"},
            "seq_num": {"sequence", "order", "number", "rank"},
            "route": {"route", "administration", "path"},
            "dose_val_rx": {"dose", "quantity", "amount", "strength"},
            "org_name": {"organism", "name", "pathogen", "species"},
            "spec_type_desc": {"specimen", "type", "source", "sample"},
            "fluid": {"specimen", "type", "source"},
            "category": {"type", "class", "concept", "domain"},
        }

        expanded = set(tokens)
        for token in list(tokens):
            expanded.update(synonym_map.get(token, set()))
            if token.endswith("id"):
                expanded.add("identifier")
            if "date" in token or "time" in token:
                expanded.update({"date", "time", "datetime"})
            expanded.update(self._field_alias_cache.get(token, []))
        return expanded

    def _schema_semantic_similarity(
        self,
        source_table: object,
        source_col: object,
        source_desc: object,
        target_table: object,
        target_col: object,
        target_desc: object,
    ) -> float:
        source_tokens = self._expand_schema_tokens(
            self._tokenize_schema_text(source_table, source_col, source_desc)
        )
        target_tokens = self._expand_schema_tokens(
            self._tokenize_schema_text(target_table, target_col, target_desc)
        )
        if not source_tokens or not target_tokens:
            return 0.0
        return len(source_tokens & target_tokens) / len(source_tokens | target_tokens)

    def _schema_embedding_similarity(
        self,
        source_table: object,
        source_col: object,
        source_desc: object,
        source_type: object,
        target_table: object,
        target_col: object,
        target_desc: object,
        target_type: object,
    ) -> float:
        source_text = self._schema_text_for_embedding(source_table, source_col, source_desc, source_type)
        target_text = self._schema_text_for_embedding(target_table, target_col, target_desc, target_type)
        if not source_text or not target_text:
            return 0.0
        matrix = self._smd_embedding_vectorizer.transform([source_text, target_text])
        return float(cosine_similarity(matrix[0], matrix[1])[0][0])

    def _alias_similarity(
        self,
        source_table: object,
        source_col: object,
        target_table: object,
        target_col: object,
    ) -> float:
        source_table_norm = self._normalize_text(source_table)
        source_col_norm = self._normalize_text(source_col)
        target_table_norm = self._normalize_text(target_table)
        target_col_norm = self._normalize_text(target_col)

        source_key = f"{source_table_norm}.{source_col_norm}"
        target_key = f"{target_table_norm}.{target_col_norm}"
        source_aliases = {
            self._normalize_text(item)
            for item in self._field_alias_cache.get(source_key, [])
            if str(item).strip()
        }
        target_aliases = {
            self._normalize_text(item)
            for item in self._field_alias_cache.get(target_key, [])
            if str(item).strip()
        }
        source_names = {source_col_norm} | source_aliases
        target_names = {target_col_norm} | target_aliases
        if not source_names or not target_names:
            return 0.0
        if source_names & target_names:
            return 1.0

        max_sim = 0.0
        for left in source_names:
            for right in target_names:
                max_sim = max(max_sim, self.similarity_metrics.string_similarity(left, right))
        return max_sim

    def _get_few_shot_examples(self, dataset_name: str, source_row: pd.Series, k: int = 2) -> str:
        examples = self._train_examples_cache.get(dataset_name)
        if examples is None:
            examples = []
            dataset = self.smd_loader.load_dataset(dataset_name, verbose=False)
            if dataset:
                train_df = dataset.get("gold_mapping_train")
                source_names = list(dataset.get("sources", {}).keys())
                if train_df is not None and not train_df.empty and len(source_names) >= 2:
                    source_df = dataset["sources"][source_names[0]]["data"]
                    target_df = dataset["sources"][source_names[1]]["data"]
                    for _, row in train_df.iterrows():
                        src = source_df[
                            (source_df["TableName"].astype(str) == str(row["source_table"]))
                            & (source_df["ColumnName"].astype(str) == str(row["source_column"]))
                        ]
                        tgt = target_df[
                            (target_df["TableName"].astype(str) == str(row["target_table"]))
                            & (target_df["ColumnName"].astype(str) == str(row["target_column"]))
                        ]
                        if not src.empty and not tgt.empty:
                            examples.append({
                                "source": src.iloc[0].to_dict(),
                                "target": tgt.iloc[0].to_dict(),
                            })
            self._train_examples_cache[dataset_name] = examples

        if not examples:
            return ""

        src_role = self._infer_schema_role(
            str(source_row.get("TableName", "")),
            str(source_row.get("ColumnName", "")),
            str(source_row.get("ColumnDesc", "")),
        )
        similar_examples = []
        for item in examples:
            ex_source = item["source"]
            ex_role = self._infer_schema_role(
                str(ex_source.get("TableName", "")),
                str(ex_source.get("ColumnName", "")),
                str(ex_source.get("ColumnDesc", "")),
            )
            if ex_role == src_role:
                similar_examples.append(item)
            if len(similar_examples) >= k:
                break
        if not similar_examples:
            similar_examples = examples[:k]

        lines = ["Examples of correct matches:"]
        for item in similar_examples:
            src = item["source"]
            tgt = item["target"]
            lines.append(
                f"- {src.get('TableName', '')}.{src.get('ColumnName', '')} -> "
                f"{tgt.get('TableName', '')}.{tgt.get('ColumnName', '')}"
            )
        return "\n".join(lines)

    def _infer_schema_role(
        self,
        table_name: object,
        column_name: object,
        description: object,
    ) -> str:
        tokens = self._expand_schema_tokens(
            self._tokenize_schema_text(table_name, column_name, description)
        )
        joined = " ".join(sorted(tokens))

        role_rules = [
            ("person_id", {"person", "patient", "subject"}),
            ("visit_occurrence_id", {"visit", "occurrence", "admission", "hadm"}),
            ("visit_detail_id", {"detail", "icustay", "stay"}),
            ("provider_id", {"provider", "caregiver", "cgid"}),
            ("gender", {"gender", "sex"}),
            ("birth_datetime", {"birth", "dob"}),
            ("death_datetime", {"death", "dod"}),
            ("datetime", {"datetime", "time", "charttime", "start", "end", "intime", "outtime"}),
            ("date", {"date", "chartdate"}),
            ("unit", {"unit", "uom"}),
            ("value_number", {"valuenum", "numeric", "number", "amount_value"}),
            ("value_text", {"value", "result", "source", "string", "text"}),
            ("concept_code", {"code", "concept", "loinc", "cpt", "icd"}),
            ("source_value", {"source", "verbatim"}),
            ("name", {"name", "title", "label", "description"}),
            ("type_concept", {"type", "category", "class"}),
        ]

        for role, keywords in role_rules:
            if keywords & tokens:
                return role

        if "id" in tokens or joined.endswith(" identifier"):
            return "identifier"
        return "unknown"

    @staticmethod
    def _schema_role_similarity(source_role: str, target_role: str) -> float:
        if not source_role or not target_role:
            return 0.0
        if source_role == target_role:
            return 1.0

        compatible_groups = [
            {"person_id", "identifier"},
            {"visit_occurrence_id", "visit_detail_id", "identifier"},
            {"provider_id", "identifier"},
            {"datetime", "date"},
            {"concept_code", "source_value", "name"},
            {"unit", "source_value"},
            {"type_concept", "name", "source_value"},
        ]
        for group in compatible_groups:
            if source_role in group and target_role in group:
                return 0.55
        semi_compatible_pairs = {
            ("birth_datetime", "date"),
            ("date", "birth_datetime"),
            ("death_datetime", "date"),
            ("date", "death_datetime"),
            ("value_number", "source_value"),
            ("source_value", "value_number"),
            ("value_text", "source_value"),
            ("source_value", "value_text"),
            ("concept_code", "type_concept"),
            ("type_concept", "concept_code"),
        }
        if (source_role, target_role) in semi_compatible_pairs:
            return 0.25
        return 0.0

    def _infer_schema_domain(
        self,
        table_name: object,
        column_name: object,
        description: object,
    ) -> str:
        tokens = self._expand_schema_tokens(
            self._tokenize_schema_text(table_name, column_name, description)
        )

        domain_rules = [
            ("person", {"person", "patient", "subject", "demographic"}),
            ("visit", {"visit", "occurrence", "admission", "encounter", "hadm"}),
            ("icu_stay", {"icustay", "stay", "detail", "icu"}),
            ("provider", {"provider", "caregiver", "cgid"}),
            ("drug", {"drug", "medication", "prescription", "dose", "route"}),
            ("procedure", {"procedure", "operation", "intervention", "treatment"}),
            ("measurement", {"measurement", "lab", "result", "test", "observation"}),
            ("unit", {"unit", "uom"}),
            ("condition", {"diagnosis", "condition", "disease", "problem"}),
            ("concept", {"concept", "code", "loinc", "cpt", "icd"}),
            ("time", {"date", "time", "datetime", "charttime", "chartdate"}),
            ("location", {"location", "caresite", "site", "ward"}),
        ]

        for domain, keywords in domain_rules:
            if keywords & tokens:
                return domain
        return "unknown"

    @staticmethod
    def _schema_domain_similarity(source_domain: str, target_domain: str) -> float:
        if not source_domain or not target_domain:
            return 0.0
        if source_domain == target_domain:
            return 1.0

        compatible_groups = [
            {"person", "provider"},
            {"visit", "icu_stay", "time"},
            {"measurement", "unit", "concept"},
            {"procedure", "visit", "concept"},
            {"drug", "measurement", "concept"},
            {"condition", "concept", "visit"},
        ]
        for group in compatible_groups:
            if source_domain in group and target_domain in group:
                return 0.5
        return 0.0

    def _generic_target_penalty(
        self,
        source_table: str,
        source_col: str,
        source_desc: str,
        target_table: str,
        target_col: str,
        target_desc: str,
    ) -> float:
        source_tokens = self._expand_schema_tokens(
            self._tokenize_schema_text(source_table, source_col, source_desc)
        )
        target_tokens = self._expand_schema_tokens(
            self._tokenize_schema_text(target_table, target_col, target_desc)
        )

        penalty = 0.0
        source_domain = self._infer_schema_domain(source_table, source_col, source_desc)
        target_domain = self._infer_schema_domain(target_table, target_col, target_desc)

        target_col_norm = self._normalize_text(target_col)
        target_table_norm = self._normalize_text(target_table)

        if "gender" in target_tokens and not ({"gender", "sex"} & source_tokens):
            penalty += 0.35
        if "unit" in target_tokens and not ({"unit", "uom"} & source_tokens):
            penalty += 0.22
        if "value source value" in target_col_norm and not ({"value", "result", "measurement", "observation"} & source_tokens):
            penalty += 0.20
        if "source value" in target_col_norm and "source" not in source_tokens and "value" not in source_tokens:
            penalty += 0.16
        if "birth datetime" in target_col_norm and not ({"birth", "dob"} & source_tokens):
            penalty += 0.42
        if "death datetime" in target_col_norm and not ({"death", "dod"} & source_tokens):
            penalty += 0.42
        if target_col_norm == "person_id" and not ({"person", "patient", "subject", "id"} & source_tokens):
            penalty += 0.26
        if target_col_norm == "person_id" and target_table_norm in {"specimen", "cost", "cohort", "cohort_definition"}:
            penalty += 0.20
        if target_col_norm in {"measurement_time", "measurement_datetime"} and source_domain not in {"measurement", "time"}:
            penalty += 0.22
        if target_col_norm == "value_as_number" and not ({"valuenum", "number", "numeric", "amount"} & source_tokens):
            penalty += 0.18
        if target_col_norm == "value_as_string" and not ({"text", "string", "comment"} & source_tokens):
            penalty += 0.18
        if target_col_norm == "unit_source_value" and "source" not in source_tokens:
            penalty += 0.12
        if target_col_norm == "gender_concept_id" and source_domain == "person" and "gender" in source_tokens:
            penalty += 0.06
        if target_table_norm == "person" and source_domain not in {"person"}:
            penalty += 0.28
        if target_table_norm == "measurement" and source_domain not in {"measurement", "unit", "concept"}:
            penalty += 0.14
        if target_table_norm == "procedure occurrence" and source_domain not in {"procedure", "visit", "concept"}:
            penalty += 0.16
        if target_table_norm == "specimen" and not ({"specimen", "sample", "microbiology", "lab"} & source_tokens):
            penalty += 0.20
        if target_table_norm == "survey_conduct" and "survey" not in source_tokens:
            penalty += 0.24
        if target_table_norm in {"cohort", "cohort_definition"} and "cohort" not in source_tokens:
            penalty += 0.24
        if target_table_norm in {"relationship", "concept_relationship"} and "relationship" not in source_tokens:
            penalty += 0.24
        if target_table_norm == "cost" and not ({"cost", "billing", "charge", "payment"} & source_tokens):
            penalty += 0.22
        if target_table_norm == "source_to_concept_map" and not ({"concept", "code", "mapping", "source", "vocabulary"} & source_tokens):
            penalty += 0.22
        if source_domain != "unknown" and target_domain != "unknown" and source_domain != target_domain:
            if self._schema_domain_similarity(source_domain, target_domain) == 0.0:
                penalty += 0.18

        return penalty

    def _target_specificity_bonus(
        self,
        source_table: str,
        source_col: str,
        source_desc: str,
        target_table: str,
        target_col: str,
        target_desc: str,
    ) -> float:
        source_tokens = self._expand_schema_tokens(
            self._tokenize_schema_text(source_table, source_col, source_desc)
        )
        target_tokens = self._expand_schema_tokens(
            self._tokenize_schema_text(target_table, target_col, target_desc)
        )
        source_role = self._infer_schema_role(source_table, source_col, source_desc)
        target_role = self._infer_schema_role(target_table, target_col, target_desc)
        source_domain = self._infer_schema_domain(source_table, source_col, source_desc)
        target_domain = self._infer_schema_domain(target_table, target_col, target_desc)
        target_col_norm = self._normalize_text(target_col)

        bonus = 0.0
        preferred_role_targets = {
            "person_id": {"person_id"},
            "provider_id": {"provider_id"},
            "gender": {"gender_source_value", "gender_concept_id"},
            "birth_datetime": {"birth_datetime", "year_of_birth"},
            "death_datetime": {"death_datetime"},
            "datetime": {"measurement_datetime", "measurement_time", "visit_start_datetime", "visit_end_datetime"},
            "date": {"measurement_date", "visit_start_date", "visit_end_date"},
            "unit": {"unit_source_value", "unit_concept_id"},
            "value_number": {"value_as_number"},
            "value_text": {"value_as_string"},
            "concept_code": {"source_concept_id", "concept_id", "value_as_concept_id"},
            "source_value": {"source_value", "source_concept_id"},
        }
        preferred_domain_tables = {
            "person": {"person"},
            "visit": {"visit_occurrence", "visit_detail"},
            "icu_stay": {"visit_detail"},
            "provider": {"provider"},
            "drug": {"drug_exposure"},
            "procedure": {"procedure_occurrence"},
            "measurement": {"measurement", "observation"},
            "condition": {"condition_occurrence", "observation"},
            "location": {"care_site", "location"},
        }

        target_table_norm = self._normalize_text(target_table)
        if target_role in preferred_role_targets.get(source_role, set()) or target_col_norm in preferred_role_targets.get(source_role, set()):
            bonus += 0.18
        if target_table_norm in preferred_domain_tables.get(source_domain, set()):
            bonus += 0.14

        if source_role == "unit" and target_col_norm in {"unit_source_value", "unit_concept_id"}:
            bonus += 0.16
            if target_col_norm == "unit_concept_id":
                bonus += 0.08
            elif target_col_norm == "unit_source_value":
                bonus -= 0.04
        if source_role == "value_number" and target_col_norm == "value_as_number":
            bonus += 0.18
        if source_role == "value_text" and target_col_norm == "value_as_string":
            bonus += 0.16
        if source_role == "gender":
            if target_col_norm == "gender_source_value":
                bonus += 0.18
            elif target_col_norm == "gender_concept_id":
                bonus += 0.06
        if source_role == "person_id" and target_col_norm == "person_id":
            bonus += 0.10
        if source_role == "visit_occurrence_id" and target_col_norm == "visit_occurrence_id":
            bonus += 0.18
        if source_role == "visit_detail_id" and target_col_norm == "visit_detail_id":
            bonus += 0.18
        if source_role == "provider_id" and target_col_norm == "provider_id":
            bonus += 0.18
        if source_role == "datetime" and "time" in target_tokens:
            bonus += 0.10
        if source_role == "birth_datetime" and "birth" in target_tokens:
            bonus += 0.22
        if source_role == "death_datetime" and "death" in target_tokens:
            bonus += 0.22
        if source_domain == "procedure" and target_table_norm == "procedure_occurrence":
            bonus += 0.18
        if source_domain == "drug" and target_table_norm == "drug_exposure":
            bonus += 0.22
        if source_domain == "measurement" and target_table_norm in {"measurement", "observation"}:
            bonus += 0.12
        if source_col == "subject_id" and target_col_norm == "person_id":
            bonus += 0.14
        if source_col == "hadm_id" and target_col_norm == "visit_occurrence_id":
            bonus += 0.22
        if source_col == "icustay_id" and target_col_norm == "visit_detail_id":
            bonus += 0.22
        if source_col == "cgid" and target_col_norm == "provider_id":
            bonus += 0.20
        if source_col in {"valueuom", "amountuom"} and target_col_norm == "unit_concept_id":
            bonus += 0.18
        if source_col == "value" and target_col_norm == "value_as_string":
            bonus += 0.18
        if source_col in {"cpt_cd", "cpt_number"} and target_table_norm == "procedure_occurrence":
            if target_col_norm in {"procedure_source_value", "procedure_concept_id"}:
                bonus += 0.18

        return bonus

    def _is_generic_target_field(self, target_table: str, target_col: str) -> bool:
        target_col_norm = self._normalize_text(target_col)
        target_table_norm = self._normalize_text(target_table)
        generic_columns = {
            "source_value",
            "value_source_value",
            "gender_source_value",
            "birth_datetime",
            "measurement_time",
            "measurement_datetime",
            "unit_source_value",
        }
        return target_col_norm in generic_columns or (
            target_table_norm in {"person", "measurement", "observation"}
            and any(token in target_col_norm for token in {"source value", "gender", "birth", "time", "unit"})
        )

    def _soft_type_compatibility(self, type1: str, type2: str) -> float:
        """宽松类型兼容判断，适配 MIMIC -> OMOP 的异构类型系统。"""
        if not type1 or not type2:
            return 0.6

        t1 = str(type1).lower().strip()
        t2 = str(type2).lower().strip()
        if not t1 or not t2:
            return 0.6
        if t1 == t2:
            return 1.0

        numeric = {
            'int', 'integer', 'bigint', 'smallint', 'numeric',
            'decimal', 'float', 'double', 'number',
            'int4', 'int8', 'float4', 'float8', 'real',
        }
        text = {
            'varchar', 'text', 'char', 'string', 'nvarchar',
            'character varying', 'character', 'bpchar', 'clob',
        }
        temporal = {
            'date', 'datetime', 'timestamp', 'time',
            'timestamp without time zone', 'timestamptz',
        }
        identifier = {'id', 'identifier', 'key'}
        boolean = {'bool', 'boolean', 'bit', 'flag'}
        concept = {'code', 'concept', 'vocab'}

        def get_group(t: str) -> str | None:
            if any(keyword in t for keyword in identifier):
                return 'identifier'
            if any(keyword in t for keyword in numeric):
                return 'numeric'
            if any(keyword in t for keyword in text):
                return 'text'
            if any(keyword in t for keyword in temporal):
                return 'temporal'
            if any(keyword in t for keyword in boolean):
                return 'boolean'
            if any(keyword in t for keyword in concept):
                return 'concept'
            return None

        g1, g2 = get_group(t1), get_group(t2)
        if g1 is not None and g1 == g2:
            return 0.85
        if g1 is None or g2 is None:
            return 0.6

        soft_compatible = {
            ('identifier', 'numeric'),
            ('numeric', 'identifier'),
            ('concept', 'identifier'),
            ('identifier', 'concept'),
            ('text', 'concept'),
            ('concept', 'text'),
            ('text', 'temporal'),
            ('temporal', 'text'),
            ('text', 'boolean'),
            ('boolean', 'text'),
            ('text', 'numeric'),
            ('numeric', 'text'),
        }
        if (g1, g2) in soft_compatible:
            return 0.6
        return 0.2

    @staticmethod
    def _normalize_prior_map(counts: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        priors: Dict[str, float] = {}
        for source_key, target_counts in counts.items():
            total = sum(target_counts.values())
            if total <= 0:
                continue
            for target_key, count in target_counts.items():
                priors[f"{source_key}::{target_key}"] = float(count) / float(total)
        return priors

    def _build_smd_prior_maps(
        self,
        dataset_name: str,
        dataset: Dict,
        source_name: str,
        target_name: str,
    ) -> Dict[str, Dict[str, float]]:
        cache_key = f"{dataset_name}::{source_name}::{target_name}"
        if cache_key in self._smd_prior_cache:
            return self._smd_prior_cache[cache_key]

        gold_mapping = dataset.get("gold_mapping_train")
        if gold_mapping is None or gold_mapping.empty:
            gold_mapping = dataset.get("gold_mapping")
        if gold_mapping is None or gold_mapping.empty:
            empty = {
                "table_pair_priors": {},
                "column_pair_priors": {},
                "role_pair_priors": {},
            }
            self._smd_prior_cache[cache_key] = empty
            return empty

        source_df = dataset["sources"][source_name]["data"]
        target_df = dataset["sources"][target_name]["data"]
        source_lookup = {
            (str(row.get("TableName", "")).strip(), str(row.get("ColumnName", "")).strip()): row
            for _, row in source_df.iterrows()
        }
        target_lookup = {
            (str(row.get("TableName", "")).strip(), str(row.get("ColumnName", "")).strip()): row
            for _, row in target_df.iterrows()
        }

        table_pair_counts: Dict[str, Dict[str, int]] = {}
        column_pair_counts: Dict[str, Dict[str, int]] = {}
        role_pair_counts: Dict[str, Dict[str, int]] = {}

        for _, row in gold_mapping.iterrows():
            src_key = (str(row.get("source_table", "")).strip(), str(row.get("source_column", "")).strip())
            tgt_key = (str(row.get("target_table", "")).strip(), str(row.get("target_column", "")).strip())
            if not src_key[0] or not src_key[1] or not tgt_key[0] or not tgt_key[1]:
                continue
            if tgt_key[0] == "0" or tgt_key[1] == "0":
                continue
            if src_key not in source_lookup or tgt_key not in target_lookup:
                continue

            src_row = source_lookup[src_key]
            tgt_row = target_lookup[tgt_key]

            src_table_norm = self._normalize_text(src_key[0])
            tgt_table_norm = self._normalize_text(tgt_key[0])
            src_col_norm = self._normalize_text(src_key[1])
            tgt_col_norm = self._normalize_text(tgt_key[1])
            src_role = self._infer_schema_role(
                src_row.get("TableName", ""),
                src_row.get("ColumnName", ""),
                src_row.get("ColumnDesc", ""),
            )
            tgt_role = self._infer_schema_role(
                tgt_row.get("TableName", ""),
                tgt_row.get("ColumnName", ""),
                tgt_row.get("ColumnDesc", ""),
            )

            table_pair_counts.setdefault(src_table_norm, {})
            table_pair_counts[src_table_norm][tgt_table_norm] = table_pair_counts[src_table_norm].get(tgt_table_norm, 0) + 1

            column_pair_counts.setdefault(src_col_norm, {})
            column_pair_counts[src_col_norm][tgt_col_norm] = column_pair_counts[src_col_norm].get(tgt_col_norm, 0) + 1

            role_pair_counts.setdefault(src_role, {})
            role_pair_counts[src_role][tgt_role] = role_pair_counts[src_role].get(tgt_role, 0) + 1

        priors = {
            "table_pair_priors": self._normalize_prior_map(table_pair_counts),
            "column_pair_priors": self._normalize_prior_map(column_pair_counts),
            "role_pair_priors": self._normalize_prior_map(role_pair_counts),
        }
        self._smd_prior_cache[cache_key] = priors
        return priors

    def _schema_type_compatibility(self, source_type: str, target_type: str) -> float:
        return self._soft_type_compatibility(source_type, target_type)

    def _extract_smd_field_pair_features(self, source_row: pd.Series, target_row: pd.Series) -> np.ndarray:
        source_table = self._normalize_text(source_row.get('TableName', ''))
        target_table = self._normalize_text(target_row.get('TableName', ''))
        source_col = self._normalize_text(source_row.get('ColumnName', ''))
        target_col = self._normalize_text(target_row.get('ColumnName', ''))
        source_desc = self._normalize_text(source_row.get('ColumnDesc', ''))
        target_desc = self._normalize_text(target_row.get('ColumnDesc', ''))
        source_type = self._normalize_text(source_row.get('ColumnType', ''))
        target_type = self._normalize_text(target_row.get('ColumnType', ''))
        source_pk = self._normalize_text(source_row.get('IsPK', ''))
        target_pk = self._normalize_text(target_row.get('IsPK', ''))
        source_fk = self._normalize_text(source_row.get('IsFK', ''))
        target_fk = self._normalize_text(target_row.get('IsFK', ''))

        combined_source = f"{source_table} {source_col} {source_desc}".strip()
        combined_target = f"{target_table} {target_col} {target_desc}".strip()
        semantic_overlap = self._schema_semantic_similarity(
            source_table,
            source_col,
            source_desc,
            target_table,
            target_col,
            target_desc,
        )
        embedding_similarity = self._schema_embedding_similarity(
            source_table,
            source_col,
            source_desc,
            source_type,
            target_table,
            target_col,
            target_desc,
            target_type,
        )
        role_similarity = self._schema_role_similarity(
            self._infer_schema_role(source_table, source_col, source_desc),
            self._infer_schema_role(target_table, target_col, target_desc),
        )
        type_compatibility = self._schema_type_compatibility(source_type, target_type)

        return np.array([
            self.similarity_metrics.string_similarity(source_col, target_col),
            self.similarity_metrics.token_overlap(source_col, target_col),
            self.similarity_metrics.string_similarity(source_table, target_table),
            self.similarity_metrics.string_similarity(source_desc, target_desc),
            self.similarity_metrics.token_overlap(source_desc, target_desc),
            1.0 if source_type == target_type and source_type else 0.0,
            type_compatibility,
            1.0 if source_pk == target_pk and source_pk else 0.0,
            1.0 if source_fk == target_fk and source_fk else 0.0,
            self.similarity_metrics.contextual_similarity(combined_source, combined_target),
            self.similarity_metrics.levenshtein_distance(combined_source, combined_target),
            semantic_overlap,
            embedding_similarity,
            role_similarity,
        ], dtype=float)

    def _prepare_gold_smd_training_data(
        self,
        dataset: Dict,
        split_role: str = 'train',
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], Dict]]:
        if split_role == 'match':
            gold_mapping = dataset.get('gold_mapping_match')
        else:
            gold_mapping = dataset.get('gold_mapping_train')
        if gold_mapping is None or gold_mapping.empty:
            gold_mapping = dataset.get('gold_mapping')
        if gold_mapping is None or gold_mapping.empty:
            return None

        source_names = list(dataset['sources'].keys())
        if len(source_names) < 2:
            return None

        source_df = dataset['sources'][source_names[0]]['data']
        target_df = dataset['sources'][source_names[1]]['data']

        source_lookup = {
            (str(row.get('TableName', '')).strip(), str(row.get('ColumnName', '')).strip()): row
            for _, row in source_df.iterrows()
        }
        target_lookup = {
            (str(row.get('TableName', '')).strip(), str(row.get('ColumnName', '')).strip()): row
            for _, row in target_df.iterrows()
        }

        positive_pairs = []
        positive_set = set()
        table_pair_counts: Dict[Tuple[str, str], int] = {}
        for _, row in gold_mapping.iterrows():
            src_key = (str(row['source_table']).strip(), str(row['source_column']).strip())
            tgt_key = (str(row['target_table']).strip(), str(row['target_column']).strip())

            if src_key not in source_lookup or tgt_key not in target_lookup:
                continue
            if tgt_key[0] == '0' or tgt_key[1] == '0':
                continue

            positive_pairs.append((src_key, tgt_key))
            positive_set.add((src_key, tgt_key))
            table_pair = (src_key[0], tgt_key[0])
            table_pair_counts[table_pair] = table_pair_counts.get(table_pair, 0) + 1

        if not positive_pairs:
            return None

        all_source_keys = list(source_lookup.keys())
        all_target_keys = list(target_lookup.keys())
        negative_candidates = []
        for src_key in all_source_keys:
            for tgt_key in all_target_keys:
                if (src_key, tgt_key) not in positive_set:
                    negative_candidates.append((src_key, tgt_key))

        rng = np.random.default_rng(42)
        negative_sample_size = min(len(negative_candidates), max(len(positive_pairs) * 2, 50))
        negative_indices = rng.choice(len(negative_candidates), size=negative_sample_size, replace=False)
        negative_pairs = [negative_candidates[idx] for idx in negative_indices]

        X = []
        y = []
        for src_key, tgt_key in positive_pairs:
            X.append(self._extract_smd_field_pair_features(source_lookup[src_key], target_lookup[tgt_key]))
            y.append(1)
        for src_key, tgt_key in negative_pairs:
            X.append(self._extract_smd_field_pair_features(source_lookup[src_key], target_lookup[tgt_key]))
            y.append(0)

        feature_names = [
            'column_name_similarity',
            'column_name_token_overlap',
            'table_name_similarity',
            'column_desc_similarity',
            'column_desc_token_overlap',
            'type_exact_match',
            'type_compatibility_soft',
            'pk_match',
            'fk_match',
            'context_similarity',
            'combined_levenshtein_similarity',
            'semantic_token_overlap',
            'embedding_similarity',
            'schema_role_similarity',
        ]

        metadata = {
            'positive_pairs': len(positive_pairs),
            'negative_pairs': len(negative_pairs),
            'source_name': source_names[0],
            'target_name': source_names[1],
            'table_pair_priors': {
                f"{src_table}::{tgt_table}": count / max(len(positive_pairs), 1)
                for (src_table, tgt_table), count in table_pair_counts.items()
            },
        }
        return np.array(X), np.array(y), feature_names, metadata
    
    def prepare_training_data(self, category: str, dataset_name: str) -> Optional[Tuple]:
        """准备训练数据"""
        logger.info(f"准备训练数据: {category}/{dataset_name}")
        
        data = self.dataset_loader.load_dataset(category, dataset_name)
        if not data:
            return None
        
        dfs = data['dataframes']
        if len(dfs) < 2:
            logger.error("数据集文件不足")
            return None
        
        # 对于有标签的数据集，优先使用train.csv和label列
        tableA = dfs[0].copy()
        tableB = dfs[1].copy()
        
        # 检测数据集类型 (SLD 或 SMD)
        data_type = self.detect_data_type(tableA, tableB)
        logger.info(f"数据集类型: {data_type}")
        
        # 检查是否有预标记的配对
        labeled_files = [d for d in data['files'].keys() 
                        if 'train' in d.lower() or 'label' in d.lower() or 'match' in d.lower()]
        
        pair_indices = None
        y = None
        
        if labeled_files and len(dfs) >= 4:
            # 使用预标记的配对
            logger.info("使用预标记的配对数据")
            
            # 尝试从train.csv获取配对关系
            for i, (file_name, file_info) in enumerate(data['files'].items()):
                if 'train' in file_name.lower() and i < len(dfs):
                    try:
                        train_pairs = dfs[i]
                        logger.info(f"从 {file_name} 获取 {len(train_pairs)} 对")
                        
                        # 提取索引和标签
                        if 'ltable_id' in train_pairs.columns and 'rtable_id' in train_pairs.columns:
                            pair_indices = train_pairs[['ltable_id', 'rtable_id']].values
                            y = train_pairs['label'].values if 'label' in train_pairs.columns else None
                            break
                    except Exception as e:
                        logger.debug(f"从 {file_name} 提取配对失败: {e}")
                        continue
        
        # 如果没有找到预标记配对，生成随机配对
        if pair_indices is None:
            n_pairs = min(len(tableA) * len(tableB), 5000)
            pair_indices = np.random.choice(len(tableA), size=n_pairs, replace=True)
            pair_indices_b = np.random.choice(len(tableB), size=n_pairs, replace=True)
            pair_indices = np.column_stack((pair_indices, pair_indices_b))
        
        # 提取特征
        logger.info(f"提取 {len(pair_indices)} 对的高级特征...")
        X = self.extract_advanced_features(tableA, tableB, pair_indices)
        
        if len(X) == 0 or len(X[0]) == 0:
            logger.error("特征提取失败")
            return None
        
        # 生成标签（如果没有预标记）
        if y is None:
            # 简单启发式：如果相似度很高，标记为匹配
            mean_sim = np.mean(X[:, 1:4], axis=1)  # 平均化前3个相似度
            y = (mean_sim > 0.7).astype(int)
            positive_ratio = np.sum(y) / len(y)
            logger.info(f"生成标签 - 正例比例: {positive_ratio:.2%}")
        
        logger.info(f"特征维度: {X.shape}")
        logger.info(f"标签分布 - 正例: {np.sum(y)}, 反例: {len(y) - np.sum(y)}")
        
        return X, y, tableA, tableB, data_type
    
    def train_smd_model(self, dataset_name: str,
                       model_type: str = 'gradient_boosting',
                       split_role: str = 'train') -> Optional[Dict]:
        """训练 SMD (仅元数据) 数据集模型"""
        logger.info(f"\n{'='*70}")
        logger.info(f"开始训练 SMD 数据集: {dataset_name} (使用 {model_type})")
        logger.info(f"{'='*70}")

        split_info = self.ensure_smd_split(dataset_name)
        
        # 加载 SMD 数据集
        dataset = self.smd_loader.load_dataset(dataset_name)
        if not dataset or len(dataset['sources']) < 2:
            logger.error(f"SMD 数据集加载失败或只有一个源")
            return None
        
        source_names = list(dataset['sources'].keys())
        logger.info(f"源数量: {len(source_names)}")

        prepared = self._prepare_gold_smd_training_data(dataset, split_role=split_role)
        if prepared is not None:
            X, y, feature_names, smd_meta = prepared
            logger.info(
                f"使用 gold mapping 构造字段级样本: 正例={smd_meta['positive_pairs']}, "
                f"反例={smd_meta['negative_pairs']}"
            )
        else:
            logger.warning("未找到 gold mapping，退回旧的 schema 级 SMD 训练逻辑")
            X = []
            y = []
            for i, source1 in enumerate(source_names):
                for source2 in source_names[i+1:]:
                    schema1 = dataset['sources'][source1]['schema']
                    schema2 = dataset['sources'][source2]['schema']
                    X.append(extract_smd_features(schema1, schema2))
                    y.append(1)

            if len(X) < 2:
                logger.error("特征提取失败，对数不足")
                return None

            X = np.array(X)
            y = np.array(y)
            feature_names = [
                'field_overlap', 'field_count_ratio', 'type_match_ratio',
                'special_char_ratio', 'length_std_sim'
            ]
            smd_meta = {
                'positive_pairs': int(np.sum(y)),
                'negative_pairs': int(len(y) - np.sum(y)),
                'source_name': source_names[0] if source_names else '',
                'target_name': source_names[1] if len(source_names) > 1 else '',
            }

        logger.info(f"特征维度: {X.shape}")
        logger.info(f"标签分布 - 正例: {np.sum(y)}, 反例: {len(y) - np.sum(y)}")

        if len(np.unique(y)) < 2:
            logger.error("SMD 标签只有单一类别，无法训练分类器")
            return None

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42, stratify=y
        )

        if model_type == 'gradient_boosting':
            final_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        else:
            final_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        
        final_model.fit(X_train, y_train)

        y_pred_train = final_model.predict(X_train)
        y_pred_test = final_model.predict(X_test)
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_test, average='binary', zero_division=0
        )
        
        results = {
            'dataset': dataset_name,
            'category': 'SMD',
            'data_type': 'SMD',
            'model_type': model_type,
            'pair_count': len(X),
            'source_count': len(source_names),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'label_distribution': {
                'positive': int(np.sum(y)),
                'negative': int(len(y) - np.sum(y))
            },
            'split_role': split_role,
            'split_summary': split_info or {},
            'smd_pair_metadata': smd_meta,
            'feature_names': feature_names,
            'feature_importance': dict(zip(feature_names, 
                                          final_model.feature_importances_.tolist()
                                          if hasattr(final_model, 'feature_importances_')
                                          else [1/len(feature_names)] * len(feature_names))),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
        }
        
        # 打印结果
        self.models[dataset_name] = final_model
        self.results[dataset_name] = results
        self._save_model_artifact(dataset_name, {
            'dataset': dataset_name,
            'data_type': 'SMD',
            'model_type': model_type,
            'model': final_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'source_name': smd_meta.get('source_name', ''),
            'target_name': smd_meta.get('target_name', ''),
            'table_pair_priors': smd_meta.get('table_pair_priors', {}),
        })
        
        return results
    
    def train_model(self, category: str, dataset_name: str,
                   model_type: str = 'gradient_boosting') -> Optional[Dict]:
        """训练改进的模型"""
        
        # 检查是否为 SMD 数据集
        if category.upper() == 'SMD':
            return self.train_smd_model(dataset_name, model_type)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"开始训练: {dataset_name} (使用 {model_type})")
        logger.info(f"{'='*70}")
        
        data = self.prepare_training_data(category, dataset_name)
        if data is None:
            return None
        
        X, y, tableA, tableB, data_type = data
        
        # 划分数据集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logger.info(f"数据分割: 训练={len(X_train)}, 验证={len(X_val)}, 测试={len(X_test)}")
        
        # 选择模型
        if model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                random_state=42,
                n_jobs=1
            )
        else:
            logger.error(f"未知的模型类型: {model_type}")
            return None
        
        # 训练
        logger.info("训练中...")
        model.fit(X_train, y_train)
        
        # 评估
        logger.info("评估中...")
        
        y_pred_train = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        
        y_pred_val = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred_val)
        
        y_pred_test = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_test, average='weighted', zero_division=0
        )
        
        # 特征重要性
        feature_importance = model.feature_importances_
        feature_names = [
            'exact_match', 'seq_similarity', 'jaccard_2gram', 'jaccard_3gram',
            'levenshtein', 'token_overlap', 'contextual', 'phonetic',
            'type_compat', 'length_ratio'
        ]
        
        results = {
            'dataset': dataset_name,
            'category': category,
            'data_type': data_type,
            'model_type': model_type,
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'test_accuracy': float(test_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'feature_importance': dict(zip(feature_names, feature_importance.tolist())),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
        }
        
        self.models[dataset_name] = model
        self.results[dataset_name] = results
        self._save_model_artifact(dataset_name, {
            'dataset': dataset_name,
            'category': category,
            'data_type': data_type,
            'model_type': model_type,
            'model': model,
            'feature_names': feature_names,
        })
        
        return results
    
    def get_all_datasets_info(self) -> Dict[str, Dict]:
        """获取所有数据集信息（SLD + SMD）
        
        返回格式:
        {
            'dataset_name': {
                'type': 'SLD' | 'SMD',
                'category': '原始类别',
                'size': '数据集大小/描述',
                'suggested_model': 'random_forest' | 'gradient_boosting'
            }
        }
        """
        all_datasets = {}
        
        # 收集 SLD 数据集
        sld_available = self.dataset_loader.list_available_datasets()
        for category, datasets in sld_available.items():
            for dataset_name in datasets:
                all_datasets[dataset_name] = {
                    'type': 'SLD',
                    'category': category,
                    'size': 'N/A',
                    'suggested_model': 'random_forest' if category == 'Structured' else 'gradient_boosting'
                }
        
        # 收集 SMD 数据集
        smd_available = self.smd_loader.list_datasets()
        for dataset_name in smd_available:
            size_desc = '~10 Schema pairs'
            counts = self._peek_smd_split_counts(dataset_name)
            gold_total = counts['gold_total']
            if gold_total:
                size_desc = f"{gold_total} gold mappings"
                if counts['train_count'] and counts['match_count']:
                    size_desc = (
                        f"{gold_total} total"
                        f" | train {counts['train_count']}"
                        f" | match {counts['match_count']}"
                    )
            all_datasets[dataset_name] = {
                'type': 'SMD',
                'category': 'SMD',
                'size': size_desc,
                'suggested_model': 'gradient_boosting'  # SMD 推荐用 GB
            }
        
        return all_datasets
    
    def build_dataset_selection_menu(self, expand_smd_splits: bool = False) -> List[Dict[str, str]]:
        all_datasets_info = self.get_all_datasets_info()
        self._menu_dataset_cache = {}

        if not all_datasets_info:
            print("[ERR] 未找到任何数据集")
            return []

        print("\n" + "="*70)
        print("[Datasets] 可用的数据集")
        print("="*70)

        entries = []
        ordered_items = sorted(
            all_datasets_info.items(),
            key=lambda item: (item[1]['category'], item[0].lower())
        )

        display_index = 1
        for dataset_name, info in ordered_items:
            if info['type'] == 'SMD' and expand_smd_splits:
                dataset = self.smd_loader.load_dataset(dataset_name, verbose=False)
                if dataset is not None:
                    self._menu_dataset_cache[dataset_name] = dataset
                train_split = dataset.get('gold_mapping_train') if dataset else None
                match_split = dataset.get('gold_mapping_match') if dataset else None
                if train_split is not None and not train_split.empty:
                    print(f"  {display_index:2d}. {dataset_name:20s}  训练集 ({len(train_split)} mappings)")
                    entries.append({
                        'display_name': f"{dataset_name} [训练集]",
                        'dataset_name': dataset_name,
                        'dataset_type': info['type'],
                        'split_role': 'train',
                    })
                    display_index += 1
                if match_split is not None and not match_split.empty:
                    print(f"  {display_index:2d}. {dataset_name:20s}  匹配集 ({len(match_split)} mappings)")
                    entries.append({
                        'display_name': f"{dataset_name} [匹配集]",
                        'dataset_name': dataset_name,
                        'dataset_type': info['type'],
                        'split_role': 'match',
                    })
                    display_index += 1
                continue

            hint = info['size'] if info['category'] == 'SMD' else info['category']
            print(f"  {display_index:2d}. {dataset_name:20s}  {hint}")
            entries.append({
                'display_name': dataset_name,
                'dataset_name': dataset_name,
                'dataset_type': info['type'],
                'split_role': 'default',
            })
            display_index += 1

        print("\n" + "="*70)
        return entries

    def print_all_datasets_menu(self) -> List[Tuple[str, str]]:
        """打印统一数据集菜单。
        
        返回: [(dataset_name, dataset_type), ...] 列表
        """
        entries = self.build_dataset_selection_menu(expand_smd_splits=False)
        return [(entry['dataset_name'], entry['dataset_type']) for entry in entries]
    
    def train_single_dataset(self, dataset_name: str, 
                            model_type: Optional[str] = None,
                            split_role: str = 'train') -> Optional[Dict]:
        """训练单个数据集，自动识别类型
        
        Args:
            dataset_name: 数据集名称
            model_type: 模型类型 (None = 自动推荐)
        """
        scene_info = self.identify_dataset_scene(dataset_name)
        
        if scene_info is None:
            logger.error(f"数据集不存在: {dataset_name}")
            return None
        
        dataset_type = scene_info['type']
        category = scene_info['category']
        suggested_model = scene_info['recommended_model']
        
        # 如果未指定模型，使用推荐模型
        if model_type is None:
            model_type = suggested_model
            logger.info(f"自动选择模型: {model_type} (推荐用于 {dataset_type})")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"正在训练数据集: {dataset_name}")
        logger.info(f"数据集类型: {dataset_type}")
        logger.info(f"所属类别: {category}")
        logger.info(f"识别场景: {scene_info['scene']}")
        logger.info(f"匹配策略: {scene_info['matching_strategy']}")
        logger.info(f"模型类型: {model_type}")
        logger.info(f"{'='*70}")
        
        # 调用相应的训练方法
        if dataset_type == 'SMD':
            return self.train_smd_model(dataset_name, model_type, split_role=split_role)
        else:
            return self.train_model(category, dataset_name, model_type)
    
    def train_multiple_datasets(self, category: str = 'Structured', 
                               max_datasets: Optional[int] = None,
                               model_type: str = 'gradient_boosting'):
        """批量训练多个数据集"""
        
        # 处理 SMD 数据集
        if category.upper() == 'SMD':
            smd_datasets = list(self.smd_loader.list_datasets().keys())
            
            if not smd_datasets:
                logger.error("未找到 SMD 数据集。请先运行 download_smd_datasets.py")
                return
            
            if max_datasets:
                smd_datasets = smd_datasets[:max_datasets]
            
            logger.info(f"准备训练 {len(smd_datasets)} 个 SMD 数据集...")
            logger.info(f"模型类型: {model_type}")
            
            for dataset_name in smd_datasets:
                try:
                    self.train_model('SMD', dataset_name, model_type)
                except Exception as e:
                    logger.error(f"训练 {dataset_name} 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            self.save_results()
            return
        
        # 处理普通数据集
        available = self.dataset_loader.list_available_datasets()
        
        if category not in available:
            logger.error(f"类别不存在: {category}")
            return
        
        datasets = available[category]
        if max_datasets:
            datasets = datasets[:max_datasets]
        
        logger.info(f"准备训练 {len(datasets)} 个数据集...")
        logger.info(f"模型类型: {model_type}")
        
        for dataset_name in datasets:
            try:
                self.train_model(category, dataset_name, model_type)
            except Exception as e:
                logger.error(f"训练 {dataset_name} 失败: {e}")
                continue
        
        self.save_results()
    
    def save_results(self, output_file: str = 'training_results_improved.json'):
        """保存训练结果"""
        logger.info(f"\n保存结果到: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info("[OK] 结果已保存")

def main():
    """主函数 - 支持统一的智能数据集选择"""
    
    trainer = ImprovedSchemaMatchingTrainer('datasets')
    
    # 检查数据集
    available = trainer.dataset_loader.list_available_datasets()
    smd_datasets = trainer.smd_loader.list_datasets()
    
    if not available and not smd_datasets:
        print("[ERR] 未找到数据集。请先运行 download_datasets.py 或 download_smd_datasets.py")
        return
    
    # 主菜单：浏览模式
    print("\n" + "="*70)
    print("Schema Matching 智能训练系统")
    print("="*70)
    print("\n选择浏览模式:")
    print("1. 混合视图 - 查看所有数据集 (推荐) ")
    print("2. 分类视图 - 按类别浏览 (传统)")
    print("3. 退出")
    
    browse_choice = input("\n请选择 (默认为 1): ").strip() or "1"
    
    if browse_choice == "3":
        print("退出")
        return
    
    # 选择模型
    print("\n" + "-"*70)
    print("选择默认模型类型 (智能体会自动推荐最优模型):")
    print("1. 自动推荐 (推荐)")
    print("2. Gradient Boosting (更稳定)")
    print("3. Random Forest (更快)")
    
    model_choice = input("\n请选择 (默认为 1): ").strip() or "1"
    
    if model_choice == "2":
        force_model = 'gradient_boosting'
        print("已选择: 强制使用 Gradient Boosting")
    elif model_choice == "3":
        force_model = 'random_forest'
        print("已选择: 强制使用 Random Forest")
    else:
        force_model = None
        print("已选择: 智能体自动推荐最优模型")
    
    # 混合视图模式
    if browse_choice == "1":
        dataset_list = trainer.print_all_datasets_menu()
        
        if not dataset_list:
            return
        
        print("\n选择数据集进行训练:")
        print("0. 训练全部")
        print("q. 返回主菜单")
        
        choice = input("\n请输入数据集编号 (默认为 0 训练全部): ").strip() or "0"
        
        if choice.lower() == 'q':
            return main()  # 回到主菜单
        
        if choice == "0":
            # 训练全部数据集
            logger.info(f"\n准备训练 {len(dataset_list)} 个数据集...")
            
            for dataset_name, dataset_type in dataset_list:
                try:
                    all_datasets = trainer.get_all_datasets_info()
                    dataset_info = all_datasets[dataset_name]
                    suggested_model = dataset_info['suggested_model']
                    
                    # 如果强制指定模型，使用强制模型；否则使用推荐模型
                    model = force_model or suggested_model
                    
                    result = trainer.train_single_dataset(dataset_name, model)
                    
                    if result:
                        logger.info(f"[OK] {dataset_name} 训练成功")
                    else:
                        logger.error(f"[ERR] {dataset_name} 训练失败")
                
                except Exception as e:
                    logger.error(f"[ERR] {dataset_name} 训练异常: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            trainer.save_results()
            
        elif choice.isdigit() and 1 <= int(choice) <= len(dataset_list):
            # 训练单个数据集
            dataset_name, dataset_type = dataset_list[int(choice) - 1]
            
            all_datasets = trainer.get_all_datasets_info()
            dataset_info = all_datasets[dataset_name]
            suggested_model = dataset_info['suggested_model']
            
            # 确认模型选择
            print(f"\n正在训练: {dataset_name} ({dataset_type})")
            if force_model:
                model = force_model
                print(f"强制使用模型: {model}")
            else:
                model = suggested_model
                print(f"智能体推荐模型: {model}")
            
            confirm = input("确认训练? (y/n, 默认为 y): ").strip().lower() or "y"
            if confirm == 'y':
                result = trainer.train_single_dataset(dataset_name, model)
                if result:
                    trainer.save_results()
                    print(f"\n[OK] 训练完成！结果已保存到 training_results_improved.json")
        else:
            print("选择无效")
    
    # 分类视图模式（传统）
    elif browse_choice == "2":
        if available:
            trainer.dataset_loader.print_available_summary()
        
        print("\n选择数据集类别:")
        categories = list(available.keys()) if available else []
        
        if smd_datasets:
            categories.append('SMD')
        
        for i, cat in enumerate(categories, 1):
            if cat == 'SMD':
                print(f"{i}. {cat} ({len(smd_datasets)} 个数据集) - 仅元数据")
            else:
                print(f"{i}. {cat} ({len(available[cat])} 个数据集)")
        
        choice = input("\n请选择 (默认为 1): ").strip() or "1"
        
        if choice.isdigit() and 1 <= int(choice) <= len(categories):
            category = categories[int(choice) - 1]
            
            if category == 'SMD':
                max_datasets = input("最多训练多少个 SMD 数据集? (默认为全部): ").strip()
            else:
                max_datasets = input("最多训练多少个数据集? (默认为全部): ").strip()
            
            max_datasets = int(max_datasets) if max_datasets else None
            
            # 使用强制模型或自动选择
            model_type = force_model or 'gradient_boosting'
            
            trainer.train_multiple_datasets(category, max_datasets, model_type)
        else:
            print("选择无效")

if __name__ == '__main__':
    main()
