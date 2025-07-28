# rank_section.py
# Ranks sections using a predefined or heuristic order

def rank_sections(sections):
    """Rank sections based on a predefined academic or report-like order."""
    preferred_order = [
        "Abstract", "Executive Summary", "Introduction", "Background",
        "Methodology", "Approach", "Key Results", "Findings", "Results",
        "Discussion", "Analysis", "Conclusion", "Summary", "References"
    ]

    section_rank = {name: i for i, name in enumerate(preferred_order)}

    # Sort sections that are in preferred_order first, then unknown ones alphabetically
    sections_sorted = sorted(
        sections,
        key=lambda sec: section_rank.get(sec, 100 + hash(sec) % 100)
    )

    return sections_sorted


# Example usage
if __name__ == "__main__":
    test_sections = ["References", "Introduction", "Conclusion", "Custom Part"]
    print(rank_sections(test_sections))
