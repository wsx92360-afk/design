"""
下载并整理 DeepMatcher / Schema Matching 数据集。

数据来源:
- https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md
- https://pages.cs.wisc.edu/~anhai/projects/schema-matching.html
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import requests


DATASETS = {
    "Structured": {
        "url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured.zip",
        "description": (
            "结构化数据集（BeerAdvo-RateBeer, iTunes-Amazon, Fodors-Zagats, "
            "DBLP-ACM, DBLP-Scholar, Amazon-Google, Walmart-Amazon）"
        ),
    },
    "Textual": {
        "url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual.zip",
        "description": "文本数据集（Abt-Buy, Company）",
    },
    "Dirty": {
        "url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty.zip",
        "description": "脏数据集（iTunes-Amazon, DBLP-ACM, DBLP-Scholar, Walmart-Amazon）",
    },
    "StructuredRaw": {
        "url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/StructuredRaw.zip",
        "description": "结构化原始数据",
    },
    "TextualRaw": {
        "url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/TextualRaw.zip",
        "description": "文本原始数据",
    },
}

DATASET_INFO = {
    "Structured": {
        "BeerAdvo-RateBeer": {"size": 450, "positive": 68, "attributes": 4, "type": "beer"},
        "iTunes-Amazon": {"size": 539, "positive": 132, "attributes": 8, "type": "music"},
        "Fodors-Zagats": {"size": 946, "positive": 110, "attributes": 6, "type": "restaurant"},
        "DBLP-ACM": {"size": 12363, "positive": 2220, "attributes": 4, "type": "citation"},
        "DBLP-Scholar": {"size": 28707, "positive": 5347, "attributes": 4, "type": "citation"},
        "Amazon-Google": {"size": 11460, "positive": 1167, "attributes": 3, "type": "software"},
        "Walmart-Amazon": {"size": 10242, "positive": 962, "attributes": 5, "type": "electronics"},
    },
    "Textual": {
        "Abt-Buy": {"size": 9575, "positive": 1028, "attributes": 3, "type": "product"},
        "Company": {"size": 112632, "positive": 28200, "attributes": 1, "type": "company"},
    },
}

SOURCE_NOTES = {
    "deepmatcher_datasets_md": "https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md",
    "schema_matching_project": "https://pages.cs.wisc.edu/~anhai/projects/schema-matching.html",
    "note": (
        "schema-matching 项目页主要是领域总览与 Illinois Semantic Integration Archive 入口；"
        "可直接批量下载并用于训练的预处理数据来自 DeepMatcher Datasets.md 中列出的 zip 包。"
    ),
}


def download_file(url: str, save_path: Path, chunk_size: int = 8192) -> None:
    print(f"\n下载: {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded_size = 0

    with save_path.open("wb") as file_obj:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            file_obj.write(chunk)
            downloaded_size += len(chunk)
            if total_size:
                progress = (downloaded_size / total_size) * 100
                print(
                    f"  进度: {progress:6.2f}% ({downloaded_size}/{total_size} bytes)",
                    end="\r",
                )

    if total_size:
        print()
    print(f"[OK] 已保存: {save_path}")


def extract_zip(zip_path: Path, extract_path: Path) -> None:
    print(f"解压缩: {zip_path} -> {extract_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print("[OK] 解压完成")


def save_metadata(base_dir: Path) -> None:
    info_file = base_dir / "dataset_info.json"
    source_file = base_dir / "dataset_sources.json"

    with info_file.open("w", encoding="utf-8") as file_obj:
        json.dump(DATASET_INFO, file_obj, ensure_ascii=False, indent=2)

    with source_file.open("w", encoding="utf-8") as file_obj:
        json.dump(SOURCE_NOTES, file_obj, ensure_ascii=False, indent=2)

    print(f"[OK] 元数据已写入: {info_file}")
    print(f"[OK] 数据来源说明已写入: {source_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 DeepMatcher / Schema Matching 数据集")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        default=["Structured", "Textual", "Dirty"],
        help="要下载的数据集分组，默认下载 Structured / Textual / Dirty",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets",
        help="输出目录，默认 ./datasets",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="保留下载后的 zip 文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.output_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DeepMatcher / Schema Matching 数据集下载")
    print("=" * 72)
    print(f"输出目录: {base_dir}")
    print(f"下载分组: {', '.join(args.datasets)}")

    for dataset_name in args.datasets:
        dataset_info = DATASETS[dataset_name]
        save_dir = base_dir / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)

        zip_path = save_dir / f"{dataset_name}.zip"
        download_file(dataset_info["url"], zip_path)
        extract_zip(zip_path, save_dir)

        if not args.keep_zip and zip_path.exists():
            zip_path.unlink()
            print(f"[OK] 已删除压缩包: {zip_path}")

    save_metadata(base_dir)
    print("=" * 72)
    print("全部完成")
    print("=" * 72)


if __name__ == "__main__":
    main()
