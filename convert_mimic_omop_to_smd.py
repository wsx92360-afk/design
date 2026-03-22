"""
将 MIMIC_2_OMOP 仓库中的 CSV schema 数据转换为项目当前可用的 SMD 数据集格式。
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


SOURCE_ROOT = Path("MIMIC_2_OMOP-main") / "data"
TARGET_ROOT = Path("datasets") / "SMD"
DATASET_NAME = "mimic_2_omop"
INVALID_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def prettify_xml(root: ET.Element) -> str:
    return ET.tostring(root, encoding="unicode")


def sanitize_tag(tag: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_:-]", "_", str(tag).strip())
    if not sanitized:
        sanitized = "field"
    if sanitized[0].isdigit():
        sanitized = f"field_{sanitized}"
    return sanitized


def build_schema_xml(rows: list[dict], root_tag: str) -> str:
    root = ET.Element(root_tag)
    for row in rows:
        record = ET.SubElement(root, "record")
        for key, value in row.items():
            child = ET.SubElement(record, sanitize_tag(key))
            text = "" if pd.isna(value) else str(value)
            child.text = INVALID_XML_RE.sub(" ", text)
    return prettify_xml(root)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ensure_repo_files() -> tuple[Path, Path, Path]:
    mimic_path = SOURCE_ROOT / "MIMIC_III_Schema.csv"
    omop_path = SOURCE_ROOT / "OMOP_Schema.csv"
    mapping_path = SOURCE_ROOT / "MIMIC_to_OMOP_Mapping.csv"

    missing = [str(path) for path in [mimic_path, omop_path, mapping_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少仓库数据文件: {missing}")

    return mimic_path, omop_path, mapping_path


def normalize_mapping(mapping_df: pd.DataFrame) -> pd.DataFrame:
    normalized = mapping_df.copy()
    normalized.columns = [col.strip() for col in normalized.columns]
    normalized = normalized.rename(
        columns={
            "SRC_ENT": "source_table",
            "SRC_ATT": "source_column",
            "TGT_ENT": "target_table",
            "TGT_ATT": "target_column",
        }
    )
    normalized = normalized[
        (normalized["target_table"].fillna("NA") != "NA")
        & (normalized["target_column"].fillna("NA") != "NA")
    ].reset_index(drop=True)
    return normalized


def create_metadata(mimic_df: pd.DataFrame, omop_df: pd.DataFrame, mapping_df: pd.DataFrame) -> dict:
    return {
        "name": DATASET_NAME,
        "type": "SMD",
        "source_repo": "https://github.com/meniData1/MIMIC_2_OMOP",
        "description": "MIMIC-III 到 OMOP 的 schema-only 映射数据集，含 gold-standard mapping。",
        "sources": {
            "mimic_iii": {
                "rows": int(len(mimic_df)),
                "tables": int(mimic_df["TableName"].nunique()),
                "columns": int(mimic_df["ColumnName"].nunique()),
            },
            "omop": {
                "rows": int(len(omop_df)),
                "tables": int(omop_df["TableName"].nunique()),
                "columns": int(omop_df["ColumnName"].nunique()),
            },
        },
        "gold_mapping_rows": int(len(mapping_df)),
    }


def main() -> None:
    mimic_path, omop_path, mapping_path = ensure_repo_files()

    mimic_df = pd.read_csv(mimic_path)
    omop_df = pd.read_csv(omop_path)
    mapping_df = normalize_mapping(pd.read_csv(mapping_path))

    if TARGET_ROOT.exists():
        shutil.rmtree(TARGET_ROOT)

    dataset_root = TARGET_ROOT / DATASET_NAME
    mimic_source_dir = dataset_root / "sources" / "mimic_iii"
    omop_source_dir = dataset_root / "sources" / "omop"
    dataset_root.mkdir(parents=True, exist_ok=True)

    mediated_schema = """<!ELEMENT schema (record*)>
<!ELEMENT record (TableName, TableDesc, ColumnName, ColumnDesc, ColumnType)>
<!ELEMENT TableName (#PCDATA)>
<!ELEMENT TableDesc (#PCDATA)>
<!ELEMENT ColumnName (#PCDATA)>
<!ELEMENT ColumnDesc (#PCDATA)>
<!ELEMENT ColumnType (#PCDATA)>
"""
    write_text(dataset_root / "mediated-schema.dtd", mediated_schema)

    mimic_xml = build_schema_xml(mimic_df.to_dict(orient="records"), "mimic_schema")
    omop_xml = build_schema_xml(omop_df.to_dict(orient="records"), "omop_schema")
    write_text(mimic_source_dir / "mimic_iii.xml", mimic_xml)
    write_text(omop_source_dir / "omop.xml", omop_xml)

    write_text(
        mimic_source_dir / "mimic_iii-schema.dtd",
        "<!ELEMENT mimic_schema (record*)>\n<!ELEMENT record ANY>\n",
    )
    write_text(
        omop_source_dir / "omop-schema.dtd",
        "<!ELEMENT omop_schema (record*)>\n<!ELEMENT record ANY>\n",
    )

    mimic_df.to_csv(mimic_source_dir / "mimic_iii_schema.csv", index=False, encoding="utf-8")
    omop_df.to_csv(omop_source_dir / "omop_schema.csv", index=False, encoding="utf-8")
    mapping_df.to_csv(dataset_root / "gold_mapping.csv", index=False, encoding="utf-8")

    metadata = create_metadata(mimic_df, omop_df, mapping_df)
    write_text(dataset_root / "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
