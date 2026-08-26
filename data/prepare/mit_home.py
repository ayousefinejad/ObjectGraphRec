"""MIT Indoor categories aligned with residential Home (Amazon Home & Kitchen)."""

# 67 MIT Indoor folder names: subset for dwelling / home scenes only.
MIT_HOME_CATEGORIES = frozenset({
    "bathroom",
    "bedroom",
    "children_room",
    "closet",
    "corridor",
    "dining_room",
    "gameroom",
    "garage",
    "kitchen",
    "livingroom",
    "nursery",
    "pantry",
    "stairscase",
    "winecellar",
})


def category_from_path(path, mit_root) -> str | None:
    """Return MIT category folder name, e.g. kitchen, from .../Images/kitchen/a.jpg."""
    parts = path.parts
    if "Images" in parts:
        i = parts.index("Images") + 1
        if i < len(parts):
            return parts[i].lower()
    try:
        rel = path.relative_to(mit_root)
    except ValueError:
        return None
    if len(rel.parts) >= 2:
        return rel.parts[0].lower()
    return None


def is_home_category(name: str | None) -> bool:
    return name is not None and name.lower() in MIT_HOME_CATEGORIES
