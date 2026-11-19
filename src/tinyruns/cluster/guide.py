from pathlib import Path

GUIDE_FILE = Path(__file__).parent.parent / "guide.md"


def guide_text() -> str:
    return GUIDE_FILE.read_text()
