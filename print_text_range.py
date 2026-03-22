from pathlib import Path
import sys


path = Path(sys.argv[1])
start = int(sys.argv[2])
end = int(sys.argv[3])

lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
for idx in range(max(1, start), min(len(lines), end) + 1):
    print(f"{idx}: {lines[idx - 1]}")
