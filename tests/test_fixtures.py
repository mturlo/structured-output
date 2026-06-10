from pathlib import Path

BRIEFS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "briefs"


def test_briefs_exist():
    briefs = sorted(BRIEFS_DIR.glob("*.md"))
    assert len(briefs) >= 5, f"Expected >=5 briefs, got {len(briefs)}"


def test_briefs_non_empty_and_have_title():
    for path in BRIEFS_DIR.glob("*.md"):
        text = path.read_text()
        assert len(text) > 200, f"{path.name} suspiciously short"
        assert text.lstrip().startswith("# "), f"{path.name} missing H1 title"
