from pathlib import Path
import sys
import zipfile
import xml.etree.ElementTree as ET


DOCX_PATH = Path(sys.argv[1])
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def main() -> int:
    if not DOCX_PATH.exists():
        print("MISSING")
        return 1

    with zipfile.ZipFile(DOCX_PATH, "r") as zf:
        xml_bytes = zf.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    paragraphs = []
    for para in root.findall(".//w:p", NS):
        texts = []
        for node in para.findall(".//w:t", NS):
            if node.text:
                texts.append(node.text)
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)

    print("\n".join(paragraphs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
