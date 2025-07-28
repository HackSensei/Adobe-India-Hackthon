# generate_output_json.py
# Final step: generates output JSON in Challenge 1B format

import json
import os
from rank_section import rank_sections
from summarize_section import summarize


def generate_output_json(pdf_id, section_lines, output_path):
    """
    Args:
        pdf_id (str): Unique PDF identifier (e.g., 'example1.pdf')
        section_lines (list of dict): [{'text': ..., 'section': ..., 'rank': ...}, ...]
        output_path (str): Output .json path
    """
    # Group by section
    section_map = {}
    for entry in section_lines:
        sec = entry["section"]
        if sec not in section_map:
            section_map[sec] = []
        section_map[sec].append(entry["text"])

    output = {
        "pdf_id": pdf_id,
        "summaries": []
    }

    ranked_sections = rank_sections(list(section_map.keys()))

    for rank, section in enumerate(ranked_sections):
        content = " ".join(section_map[section])
        summary = summarize(content)
        output["summaries"].append({
            "section": section,
            "summary": summary,
            "rank": rank + 1
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Output JSON written to {output_path}")


# Example usage
if __name__ == "__main__":
    example_lines = [
        {"text": "This paper explores...", "section": "Abstract"},
        {"text": "We propose a novel method...", "section": "Abstract"},
        {"text": "1. Introduction", "section": "Introduction"},
        {"text": "Our method builds upon prior work...", "section": "Introduction"}
    ]

    generate_output_json("example1.pdf", example_lines, "outputs/example1.json")
