import json
from pathlib import Path

import pandas as pd
import requests


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3.1:latest"


def generate_aliases_for_field(table: str, column: str, desc: str, col_type: str) -> list[str]:
    prompt = (
        "You are a medical data expert familiar with MIMIC-III and OMOP CDM terminology.\n"
        "Generate 4 to 6 alternative names for the following source field that would match OMOP-style naming.\n"
        "Return only a JSON array of strings.\n\n"
        f"Table: {table}\n"
        f"Column: {column}\n"
        f"Type: {col_type}\n"
        f"Description: {desc}"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = str(resp.json().get("response", "")).strip()
        return json.loads(raw) if raw else []
    except Exception:
        return []


def main() -> int:
    source_csv = Path("datasets/SMD/mimic_2_omop/sources/mimic_iii/mimic_iii_schema.csv")
    if not source_csv.exists():
        print(f"missing source schema: {source_csv}")
        return 1

    df = pd.read_csv(source_csv)
    alias_map: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        table = str(row.get("TableName", "")).strip()
        column = str(row.get("ColumnName", "")).strip()
        key = f"{table.lower()}.{column.lower()}"
        alias_map[key] = generate_aliases_for_field(
            table,
            column,
            str(row.get("ColumnDesc", "")),
            str(row.get("ColumnType", "")),
        )
        print(f"{key}: {alias_map[key]}")

    Path("mimic_field_aliases.json").write_text(
        json.dumps(alias_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("saved -> mimic_field_aliases.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
