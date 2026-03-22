"""
Schema Matching 数据集处理和加载工具
支持从本地CSV文件加载和预处理数据
"""

import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class SchemaMatchingDataset:
    """Schema Matching 数据集处理类"""
    
    def __init__(self, dataset_dir: str = 'datasets'):
        self.dataset_dir = Path(dataset_dir)
        self.datasets = {}
        self.dataset_info = {}
        self._load_dataset_info()
    
    def _load_dataset_info(self):
        """加载数据集信息"""
        info_file = self.dataset_dir / 'dataset_info.json'
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                self.dataset_info = json.load(f)
    
    def list_available_datasets(self) -> Dict[str, List[str]]:
        """列出所有可用的数据集"""
        available = {}
        
        for category in ['Structured', 'Textual', 'Dirty']:
            category_dir = self.dataset_dir / category
            if category_dir.exists():
                # 处理嵌套目录问题（某些下载可能创建了 Structured/Structured 等嵌套结构）
                nested_dir = category_dir / category
                if nested_dir.exists():
                    search_dir = nested_dir
                else:
                    search_dir = category_dir
                
                datasets = [d.name for d in search_dir.iterdir() 
                           if d.is_dir() and d.name != '__pycache__']
                available[category] = sorted(datasets)
        
        return available
    
    def load_dataset(self, category: str, dataset_name: str, verbose: bool = False) -> Optional[pd.DataFrame]:
        """
        加载单个数据集
        
        Args:
            category: 数据集类别 (Structured, Textual, Dirty)
            dataset_name: 数据集名称 (如 'BeerAdvo-RateBeer')
        
        Returns:
            合并后的DataFrame或None
        """
        # 处理嵌套目录问题
        category_dir = self.dataset_dir / category
        nested_dir = category_dir / category
        
        if nested_dir.exists():
            dataset_path = nested_dir / dataset_name
        else:
            dataset_path = category_dir / dataset_name
        
        if not dataset_path.exists():
            if verbose:
                logger.error(f"数据集不存在: {dataset_path}")
            return None
        
        try:
            # 查找CSV文件
            csv_files = list(dataset_path.glob('*.csv'))
            
            if not csv_files:
                logger.error(f"在 {dataset_path} 中未找到CSV文件")
                return None
            
            # 加载文件
            dfs = []
            files_info = {}
            
            for csv_file in sorted(csv_files):
                file_name = csv_file.name
                if verbose:
                    logger.info(f"加载: {file_name}")
                
                df = pd.read_csv(csv_file)
                dfs.append(df)
                files_info[file_name] = {
                    'rows': len(df),
                    'columns': list(df.columns)
                }
            
            # 合并所有表
            combined_data = {
                'category': category,
                'dataset_name': dataset_name,
                'files': files_info,
                'dataframes': dfs
            }
            
            if verbose:
                logger.info(f"✓ 成功加载数据集: {dataset_name}")
            return combined_data
            
        except Exception as e:
            if verbose:
                logger.error(f"加载数据集失败: {e}")
            return None
    
    def get_dataset_statistics(self, category: str, dataset_name: str) -> Optional[Dict]:
        """获取数据集统计信息"""
        if category in self.dataset_info and dataset_name in self.dataset_info[category]:
            return self.dataset_info[category][dataset_name]
        
        return None
    
    def print_available_summary(self):
        """打印可用数据集摘要"""
        available = self.list_available_datasets()
        
        print("\n" + "=" * 70)
        print("可用的 Schema Matching 数据集")
        print("=" * 70)
        
        for category, datasets in available.items():
            print(f"\n【{category}】数据集 ({len(datasets)} 个):")
            
            if category in self.dataset_info:
                for dataset_name in sorted(datasets):
                    if dataset_name in self.dataset_info[category]:
                        info = self.dataset_info[category][dataset_name]
                        print(f"  • {dataset_name}")
                        print(f"    - 大小: {info.get('size', 'N/A')} 对")
                        print(f"    - 正例: {info.get('positive', 'N/A')} 对")
                        print(f"    - 属性: {info.get('attributes', 'N/A')} 个")
            else:
                for dataset_name in sorted(datasets):
                    print(f"  • {dataset_name}")
        
        print("\n" + "=" * 70)

class DataPreprocessor:
    """数据预处理工具"""
    
    @staticmethod
    def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
        """规范化数据"""
        df_normalized = df.copy()
        
        # 处理缺失值
        for col in df_normalized.select_dtypes(include=['object']).columns:
            df_normalized[col] = df_normalized[col].fillna('').astype(str)
        
        # 转换为小写
        for col in df_normalized.select_dtypes(include=['object']).columns:
            df_normalized[col] = df_normalized[col].str.lower()
        
        return df_normalized
    
    @staticmethod
    def create_training_split(df: pd.DataFrame, 
                            train_ratio: float = 0.6,
                            val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        划分训练、验证、测试集
        
        Args:
            df: 输入数据框
            train_ratio: 训练集比例 (默认 0.6)
            val_ratio: 验证集比例 (默认 0.2)
        
        Returns:
            (train_df, val_df, test_df)
        """
        # 确保比例有效
        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < 0:
            raise ValueError(f"训练集和验证集比例之和不能超过1.0")
        
        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        logger.info(f"数据集划分: 训练={len(train_df)} ({train_ratio*100:.0f}%), "
                   f"验证={len(val_df)} ({val_ratio*100:.0f}%), "
                   f"测试={len(test_df)} ({test_ratio*100:.0f}%)")
        
        return train_df, val_df, test_df

def main():
    """演示用法"""
    
    # 初始化数据加载器
    loader = SchemaMatchingDataset('datasets')
    
    # 显示可用数据集
    loader.print_available_summary()
    
    # 加载示例数据集
    print("\n尝试加载数据集示例...")
    available = loader.list_available_datasets()
    
    if available:
        first_category = list(available.keys())[0]
        if available[first_category]:
            first_dataset = available[first_category][0]
            print(f"\n加载: {first_category}/{first_dataset}")
            
            data = loader.load_dataset(first_category, first_dataset)
            if data:
                print(f"\n数据集信息:")
                print(f"  类别: {data['category']}")
                print(f"  名称: {data['dataset_name']}")
                print(f"\n包含的文件:")
                for file_name, file_info in data['files'].items():
                    print(f"  • {file_name}")
                    print(f"    - 行数: {file_info['rows']}")
                    print(f"    - 列: {file_info['columns']}")
                
                # 获取统计信息
                stats = loader.get_dataset_statistics(first_category, first_dataset)
                if stats:
                    print(f"\n数据集统计:")
                    print(f"  总样本对数: {stats.get('size', 'N/A')}")
                    print(f"  正例对数: {stats.get('positive', 'N/A')}")
                    print(f"  属性数: {stats.get('attributes', 'N/A')}")

if __name__ == '__main__':
    main()
