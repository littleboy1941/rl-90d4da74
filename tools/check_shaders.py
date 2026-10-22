"""Extract the Metal sources embedded in Swift string literals.

    python3 tools/check_shaders.py <swift dir> <out dir>

Every triple-quoted Swift string that contains `#include <metal_stdlib>`
becomes <out dir>/<file>_<n>.metal, ready for `xcrun metal -c`.
"""
import re
import sys
from pathlib import Path


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for swift in sorted(src.glob("*.swift")):
        text = swift.read_text(encoding="utf-8")
        for n, body in enumerate(re.findall(r'"""\n(.*?)\n\s*"""', text, re.S)):
            if "#include <metal_stdlib>" not in body:
                continue
            lines = body.splitlines()
            indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
            path = out / f"{swift.stem}_{n}.metal"
            path.write_text("\n".join(line[indent:] for line in lines) + "\n", encoding="utf-8")
            print(path)
            count += 1
    if count == 0:
        sys.exit("no Metal sources found")


if __name__ == "__main__":
    main()
