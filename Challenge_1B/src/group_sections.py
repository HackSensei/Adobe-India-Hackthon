# group_sections.py
# Groups section title candidates into structured hierarchy

import json
import re
from collections import defaultdict
import os

INPUT_PATH = "data/section_candidates.json"
OUTPUT_PATH = "data/final_outline.json"

# Simple rule-based heading levels
HEADING_PATTERNS = [
    (re.compile(r"^\d+\.\d+\."), "H3"),
    (re.compile(r"^\d+\."), "H2"),
    (re.compile(r"^[A-Z][A-Z\s]+$"), "H1")
]


def classify_heading_level(title):
    for pattern, level in HEADING_PATTERNS:
        if pattern.match(title.strip()):
            return level
    if len(title.split()) <= 3:
        return "H2"
    return "H3"


def group_sections(candidates):
    outlines = defaultdict(list)

    for cand in candidates:
        doc = cand["document"]
        section = {
            "section_title": cand["section_title"].strip(),
            "page_number": cand["page_number"],
            "heading_level": classify_heading_level(cand["section_title"]),
            "confidence": cand["confidence"]
        }
        outlines[doc].append(section)

    # sort sections by page number then title position
    for doc in outlines:
        outlines[doc] = sorted(outlines[doc], key=lambda x: (x["page_number"]))

    return outlines


def main():
    with open(INPUT_PATH, "r") as f:
        candidates = json.load(f)

    outlines = group_sections(candidates)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(outlines, f, indent=2)

    print(f"✅ Grouped {len(candidates)} titles → structured outline saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
