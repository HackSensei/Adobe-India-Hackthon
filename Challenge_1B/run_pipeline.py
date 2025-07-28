# offline_pipeline.py
# Complete offline pipeline to process PDFs into section outline JSON

import os
import sys
import subprocess
import json

sys.path.append(os.path.abspath("src"))

from extract_text_from_pdfs import extract_pdf_lines
from classify_sections import classify_titles
from group_sections import group_sections


def run_pipeline(input_folder):
    os.makedirs("data", exist_ok=True)
    print("[1/3] Extracting text from PDFs...")
    extract_pdf_lines(input_folder, output_path="data/parsed_lines.json")

    print("[2/3] Classifying section title candidates...")
    classify_titles("data/parsed_lines.json", model_path="models/distilbert_section_classifier")

    print("[3/3] Grouping sections by hierarchy...")
    with open("data/section_candidates.json") as f:
        section_candidates = json.load(f)
    outlines = group_sections(section_candidates)

    with open("data/final_outline.json", "w") as f:
        json.dump(outlines, f, indent=2)

    print("\nPipeline complete. Output saved to data/final_outline.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", required=True, help="Path to folder with PDF files")
    args = parser.parse_args()

    run_pipeline(args.pdf_dir)
