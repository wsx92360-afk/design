"""
SMD 数据集加载器
用于处理基于 XML 的 schema-only 数据集，并可选读取金标准映射。
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class SMDDatasetLoader:
    """加载 SMD (Schema with only MetaData) 数据集"""
    
    def __init__(self, dataset_dir: str = 'datasets/SMD'):
        self.dataset_dir = Path(dataset_dir)
        self.datasets = {}
        self._discover_datasets()
    
    def _discover_datasets(self):
        """发现可用的数据集"""
        if not self.dataset_dir.exists():
            logger.warning(f"数据集目录不存在: {self.dataset_dir}")
            return
        
        for dataset_path in self.dataset_dir.iterdir():
            if dataset_path.is_dir() and (dataset_path / 'sources').exists():
                dataset_name = dataset_path.name
                self.datasets[dataset_name] = dataset_path
                logger.info(f"发现 SMD 数据集: {dataset_name}")
    
    def list_datasets(self) -> Dict[str, Path]:
        """列出所有可用的数据集"""
        return self.datasets.copy()
    
    def parse_xml(self, xml_file: str, verbose: bool = False) -> Tuple[List[str], List[Dict]]:
        """
        解析 XML 文件
        
        返回:
            (字段名列表, 数据记录列表)
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            records = []
            field_names = set()
            
            # 遍历所有记录
            for record in root.findall('.//record') or root.findall('.//entry') or [root]:
                record_dict = {}
                for child in record:
                    # 使用标签名作为字段名
                    field_name = child.tag
                    field_names.add(field_name)
                    # 提取文本值
                    text_value = child.text or ''
                    record_dict[field_name] = text_value
                
                if record_dict:
                    records.append(record_dict)
            
            # 如果没有找到记录，将根元素作为单个记录
            if not records:
                record_dict = {}
                for child in root:
                    field_names.add(child.tag)
                    record_dict[child.tag] = child.text or ''
                if record_dict:
                    records.append(record_dict)
            
            field_list = sorted(list(field_names))
            if verbose:
                logger.info(f"解析 {len(records)} 条记录，{len(field_list)} 个字段")
            
            return field_list, records
        
        except Exception as e:
            logger.error(f"解析 XML 失败: {e}")
            return [], []
    
    def extract_schema_from_xml(self, xml_file: str, verbose: bool = False) -> Dict:
        """
        从 XML 提取 schema 信息
        
        返回:
            {
                'fields': [字段名列表],
                'types': {字段名: 推断类型},
                'sample_values': {字段名: 示例值},
                'statistics': {字段名: 统计信息}
            }
        """
        field_names, records = self.parse_xml(xml_file, verbose=verbose)
        
        if not records or not field_names:
            return {'fields': [], 'types': {}, 'sample_values': {}, 'statistics': {}}
        
        # 转换为 DataFrame 便于分析
        df = pd.DataFrame(records)
        
        schema = {
            'fields': field_names,
            'types': {},
            'sample_values': {},
            'statistics': {}
        }
        
        # 推断每个字段的类型和统计信息
        for field in field_names:
            if field in df.columns:
                values = df[field].dropna().astype(str).str.strip()
                
                # 推断类型
                try:
                    # 尝试转换为数字
                    numeric_values = pd.to_numeric(values, errors='coerce')
                    if numeric_values.notna().sum() / len(values) > 0.8:
                        field_type = 'numeric'
                    else:
                        field_type = 'string'
                except:
                    field_type = 'string'
                
                schema['types'][field] = field_type
                
                # 样本值
                non_empty = values[values != '']
                if len(non_empty) > 0:
                    schema['sample_values'][field] = non_empty.iloc[0]
                
                # 统计信息
                schema['statistics'][field] = {
                    'non_null_count': len(non_empty),
                    'total_count': len(values),
                    'unique_count': values.nunique(),
                    'avg_length': values.str.len().mean()
                }
        
        return schema
    
    def load_dataset(self, dataset_name: str, verbose: bool = False) -> Optional[Dict]:
        """
        加载完整的 SMD 数据集
        
        返回:
            {
                'name': 数据集名称,
                'sources': {源名: {schema, data}},
                'mediated_schema': 中介 schema,
                'data_type': 'SMD'
            }
        """
        if dataset_name not in self.datasets:
            if verbose:
                logger.error(f"数据集不存在: {dataset_name}")
            return None
        
        dataset_path = self.datasets[dataset_name]
        sources_dir = dataset_path / 'sources'
        
        dataset_info = {
            'name': dataset_name,
            'sources': {},
            'data_type': 'SMD',
            'files': {},
            'gold_mapping': None,
            'gold_mapping_train': None,
            'gold_mapping_match': None,
        }
        
        # 加载 mediated schema
        mediated_file = dataset_path / 'mediated-schema.dtd'
        if mediated_file.exists():
            try:
                with open(mediated_file, 'r', encoding='utf-8') as f:
                    dataset_info['mediated_schema'] = f.read()
            except:
                dataset_info['mediated_schema'] = None

        gold_mapping_file = dataset_path / 'gold_mapping.csv'
        if gold_mapping_file.exists():
            try:
                dataset_info['gold_mapping'] = pd.read_csv(gold_mapping_file)
                dataset_info['files'][str(gold_mapping_file)] = {
                    'type': 'gold_mapping',
                    'source': 'gold_standard'
                }
            except Exception as e:
                if verbose:
                    logger.warning(f"读取 gold mapping 失败: {e}")

        gold_mapping_train_file = dataset_path / 'gold_mapping_train.csv'
        if gold_mapping_train_file.exists():
            try:
                dataset_info['gold_mapping_train'] = pd.read_csv(gold_mapping_train_file)
                dataset_info['files'][str(gold_mapping_train_file)] = {
                    'type': 'gold_mapping_train',
                    'source': 'gold_standard_train'
                }
            except Exception as e:
                if verbose:
                    logger.warning(f"读取 gold mapping train 失败: {e}")

        gold_mapping_match_file = dataset_path / 'gold_mapping_match.csv'
        if gold_mapping_match_file.exists():
            try:
                dataset_info['gold_mapping_match'] = pd.read_csv(gold_mapping_match_file)
                dataset_info['files'][str(gold_mapping_match_file)] = {
                    'type': 'gold_mapping_match',
                    'source': 'gold_standard_match'
                }
            except Exception as e:
                if verbose:
                    logger.warning(f"读取 gold mapping match 失败: {e}")
        
        # 加载各源的数据
        for source_dir in sources_dir.iterdir():
            if source_dir.is_dir():
                source_name = source_dir.name
                
                # 查找 XML 文件
                xml_files = list(source_dir.glob('*.xml'))
                if not xml_files:
                    # 如果没有 XML 文件，尝试查找其他格式
                    if verbose:
                        logger.debug(f"源 {source_name} 中未找到 .xml 文件")
                    continue
                
                xml_file = xml_files[0]
                
                # 提取 schema
                schema = self.extract_schema_from_xml(str(xml_file), verbose=verbose)
                
                # 解析数据
                field_names, records = self.parse_xml(str(xml_file), verbose=verbose)
                
                # 转换为 DataFrame
                df = pd.DataFrame(records) if records else pd.DataFrame()
                
                dataset_info['sources'][source_name] = {
                    'schema': schema,
                    'data': df,
                    'field_names': field_names,
                    'record_count': len(records)
                }
                
                dataset_info['files'][str(xml_file)] = {
                    'type': 'data',
                    'source': source_name
                }
                
                if verbose:
                    logger.info(f"  源 {source_name}: {len(records)} 条记录, {len(field_names)} 个字段")
        
        if not dataset_info['sources']:
            if verbose:
                logger.error(f"未能加载任何源数据")
            return None
        
        if verbose:
            logger.info(f"✓ 数据集 {dataset_name} 加载完成: {len(dataset_info['sources'])} 个源")

        return dataset_info
    
    def extract_schema_pairs(self, dataset_name: str) -> List[Tuple[Dict, Dict, int]]:
        """
        提取所有的 schema 配对用于训练
        
        返回:
            [(schema1, schema2, match_score), ...]
            match_score: 0 (不匹配) 或 1 (匹配)
        """
        dataset = self.load_dataset(dataset_name)
        if not dataset:
            return []
        
        sources = dataset['sources']
        source_names = list(sources.keys())
        
        pairs = []
        
        # 生成所有的 schema 配对
        for i, source1 in enumerate(source_names):
            for source2 in source_names[i+1:]:
                schema1 = sources[source1]['schema']
                schema2 = sources[source2]['schema']
                
                # 由于都是同一个数据集的不同源，默认标记为匹配
                pairs.append((schema1, schema2, 1))
        
        return pairs


def extract_smd_features(schema1: Dict, schema2: Dict) -> np.ndarray:
    """
    为 SMD 数据集提取特征
    
    由于 SMD 没有实例数据，只能基于 schema 名称、字段类型等
    """
    features = []
    
    fields1 = set(schema1.get('fields', []))
    fields2 = set(schema2.get('fields', []))
    
    # 特征1: 字段名的重叠比例
    if len(fields1 | fields2) > 0:
        field_overlap = len(fields1 & fields2) / len(fields1 | fields2)
    else:
        field_overlap = 0.0
    features.append(field_overlap)
    
    # 特征2: 字段个数比例
    if max(len(fields1), len(fields2)) > 0:
        field_count_ratio = min(len(fields1), len(fields2)) / max(len(fields1), len(fields2))
    else:
        field_count_ratio = 0.0
    features.append(field_count_ratio)
    
    # 特征3: 类型匹配程度
    types1 = schema1.get('types', {})
    types2 = schema2.get('types', {})
    common_type_match = 0
    for field in fields1 & fields2:
        if types1.get(field) == types2.get(field):
            common_type_match += 1
    type_match_ratio = common_type_match / len(fields1 & fields2) if len(fields1 & fields2) > 0 else 0.0
    features.append(type_match_ratio)
    
    # 特征4: 特殊字符相似度（特征之一）
    common_special_chars = 0
    all_special_chars = 0
    
    for field in fields1:
        for char in ['_', '-', '.', '@', '#']:
            if char in field:
                all_special_chars += 1
                if char in ''.join(fields2):
                    common_special_chars += 1
    
    special_char_ratio = common_special_chars / max(all_special_chars, 1)
    features.append(special_char_ratio)
    
    # 特征5: 字段名长度标准差相似度
    lengths1 = [len(f) for f in fields1]
    lengths2 = [len(f) for f in fields2]
    
    if lengths1 and lengths2:
        std1 = np.std(lengths1) if len(lengths1) > 1 else 0
        std2 = np.std(lengths2) if len(lengths2) > 1 else 0
        length_std_sim = 1 - abs(std1 - std2) / (max(std1, std2) + 0.1)
    else:
        length_std_sim = 0.0
    
    features.append(length_std_sim)
    
    return np.array(features)


if __name__ == '__main__':
    # 测试
    loader = SMDDatasetLoader('datasets/SMD')
    
    print("可用的 SMD 数据集:")
    for name in loader.list_datasets():
        print(f"  • {name}")
    
    # 尝试加载第一个数据集
    datasets = loader.list_datasets()
    if datasets:
        first_dataset = list(datasets.keys())[0]
        print(f"\n加载 {first_dataset}...")
        dataset = loader.load_dataset(first_dataset)
        if dataset:
            print(f"✓ 成功加载！")
            print(f"  数据集类型: {dataset['data_type']}")
            print(f"  源数量: {len(dataset['sources'])}")
            for source_name, source_info in dataset['sources'].items():
                print(f"    - {source_name}: {source_info['record_count']} 条记录")
