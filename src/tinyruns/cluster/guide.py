from pathlib import Path

SKILL_FILE = Path(__file__).parent.parent / "skill" / "SKILL.md"
FRONT_MATTER_FENCE = "---\n"


def guide_text() -> str:
    _, _front_matter, guide = SKILL_FILE.read_text().split(FRONT_MATTER_FENCE, 2)
    return guide.lstrip("\n")
