"""Checks report/report.md and README.md against the writing rules: banned terms, em dashes, sentences of 35 words or more."""
import re
import sys

BANNED = ["strategic", "robust", "comprehensive", "compelling", "key", "landscape", "framework", "roadmap", "leverage", "unlock",
          "seamless", "holistic", "nuanced", "best-in-class", "canonical", "it is important to note"]
FIGURATIVE = ["tax", "floor", "ceiling", "moat", "lever", "wedge", "ladder", "playbook", "signal", "tell", "headwind", "tailwind",
              "killer app", "rounding error"]


def sentences(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    out = []
    for line in text.splitlines():
        if line.startswith("|") or line.startswith("#") or line.strip() in ("", "---"):
            continue
        if line.startswith("!["):
            line = line[2:line.index("](")]
        line = re.sub(r"^\s*(\d+\.|-)\s+", "", line)
        line = re.sub(r"`[^`]*`", "CODE", line)
        line = re.sub(r"\$[^$]*\$", "MATH", line)
        line = line.replace("**", "")
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", line) if s.strip()]
    return out


bad = 0
for path in sys.argv[1:]:
    raw = open(path, encoding="utf-8").read()
    for ch, name in [("—", "em dash"), ("–", "en dash")]:
        for i, line in enumerate(raw.splitlines(), 1):
            if ch in line:
                print(f"{path}:{i}: {name}"); bad += 1
    for i, line in enumerate(raw.splitlines(), 1):
        low = re.sub(r"`[^`]*`", "", line.lower())
        for w in BANNED + FIGURATIVE:
            if re.search(r"(?<![a-z-])" + re.escape(w) + r"(?![a-z])", low):
                print(f"{path}:{i}: term '{w}': {line[:90]}"); bad += 1
    for s in sentences(raw):
        n = len(s.split())
        if n >= 35:
            print(f"{path}: {n} words: {s}\n"); bad += 1
print("issues:", bad)
