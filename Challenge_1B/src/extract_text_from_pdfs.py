# extract_text_from_pdfs.py
# Parses PDFs into structured lines with font size, position, bold, etc.

import pdfplumber
import os
import json

INPUT_DIR = "data/pdfs"
OUTPUT_PATH = "data/parsed_lines.json"

# Heuristic to detect bold
BOLD_KEYWORDS = ["Bold", "bold"]

def extract_pdf_lines(pdf_path):
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            words = page.extract_words(extra_attrs=["fontname", "size", "x0", "top"])
            line_map = {}
            for word in words:
                key = round(word['top'], 1)  # group by vertical position
                if key not in line_map:
                    line_map[key] = []
                line_map[key].append(word)

            for key in sorted(line_map):
                line_words = line_map[key]
                text = " ".join(w["text"] for w in line_words)
                font_size = max(w["size"] for w in line_words)
                font_pos = (min(w["x0"] for w in line_words), key)
                bold = any(any(b in w["fontname"] for b in BOLD_KEYWORDS) for w in line_words)
                alignment = "center" if abs(page.width/2 - sum(w['x0'] for w in line_words)/len(line_words)) < 50 else "left"

                lines.append({
                    "text": text,
                    "font_size": font_size,
                    "font_position": font_pos,
                    "bold": bold,
                    "alignment": alignment,
                    "page_number": page_num + 1,
                    "source_pdf": os.path.basename(pdf_path)
                })
    return lines

def main():
    all_lines = []
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(INPUT_DIR, file)
            print(f"📄 Parsing {file}...")
            lines = extract_pdf_lines(path)
            all_lines.extend(lines)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_lines, f, indent=2)
    print(f"Extracted lines saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
