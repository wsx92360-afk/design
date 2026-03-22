from pathlib import Path
import sys


PDF_PATH = Path(sys.argv[1])


def extract_with_pypdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def extract_with_pymupdf(path: Path) -> str:
    import fitz

    doc = fitz.open(str(path))
    parts = []
    for page in doc:
        parts.append(page.get_text() or "")
    return "\n".join(parts)


def main() -> int:
    if not PDF_PATH.exists():
        print("MISSING")
        return 1

    errors = []
    for fn in (extract_with_pypdf, extract_with_pymupdf):
        try:
            text = fn(PDF_PATH)
            if text.strip():
                print(text)
                return 0
        except Exception as exc:
            errors.append(f"{fn.__name__}: {exc}")

    print("EXTRACT_FAILED")
    for err in errors:
        print(err)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
