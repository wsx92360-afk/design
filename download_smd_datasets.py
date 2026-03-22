"""
下载并安装 MIMIC_2_OMOP 作为当前项目唯一的 SMD 数据集。
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

from convert_mimic_omop_to_smd import main as convert_main


REPO_ZIP_URL = "https://codeload.github.com/meniData1/MIMIC_2_OMOP/zip/refs/heads/main"
ZIP_PATH = Path("MIMIC_2_OMOP-main.zip")
EXTRACTED_DIR = Path("MIMIC_2_OMOP-main")


def download_repo() -> None:
    with urlopen(REPO_ZIP_URL, timeout=60) as response:
        ZIP_PATH.write_bytes(response.read())
    print(f"[OK] 已下载: {ZIP_PATH}")


def extract_repo() -> None:
    if EXTRACTED_DIR.exists():
        shutil.rmtree(EXTRACTED_DIR)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(Path("."))
    print(f"[OK] 已解压: {EXTRACTED_DIR}")


def main() -> None:
    download_repo()
    extract_repo()
    convert_main()
    print("[OK] 当前 SMD 数据集已切换为 mimic_2_omop")


if __name__ == "__main__":
    main()
