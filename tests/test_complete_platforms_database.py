import json
from pathlib import Path


def test_complete_platforms_database_exists():
    p = Path("monetization-system/data/complete_platforms_database.json")
    assert p.exists(), f"Missing {p}"


def test_structure_and_counts():
    p = Path("monetization-system/data/complete_platforms_database.json")
    data = json.loads(p.read_text())

    assert "platform_categories" in data
    assert "total_platforms" in data

    categories = data["platform_categories"]

    total_from_counts = sum(c.get("count", 0) for c in categories.values())
    assert total_from_counts == data["total_platforms"], (
        f"total_platforms ({data['total_platforms']}) != sum of category counts ({total_from_counts})"
    )

    # Verify platform list lengths match counts and uniqueness
    all_platforms = []
    for name, c in categories.items():
        platforms = c.get("platforms", [])
        assert len(platforms) == c.get("count", len(platforms))
        all_platforms.extend(platforms)

    assert len(all_platforms) == data["total_platforms"]
    assert len(set(all_platforms)) == len(all_platforms), "Platform names are not unique"
