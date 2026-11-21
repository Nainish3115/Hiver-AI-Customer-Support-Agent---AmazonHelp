"""Regression guards for the lucide-react 1.74 upgrade.

The 1.74 release renamed the ``centerline`` icon to ``axis`` and no longer
exports the unused ``Github`` brand icon. The frontend does not use
``centerline``, so the upgrade needs to move the dependency pin and keep the
component imports limited to icons that still exist.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def _version_tuple(value: str) -> tuple[int, int, int]:
    """Return the first three numeric components of a semver/range string."""
    parts = [int(part) for part in re.findall(r"\d+", value)[:3]]
    return tuple(parts + [0] * (3 - len(parts)))


def test_lucide_react_1_74_upgrade_is_complete() -> None:
    package = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))
    declared = package["dependencies"]["lucide-react"]
    assert _version_tuple(declared) >= (1, 74, 0), (
        f"lucide-react must be pinned to >= 1.74.0 after the centerline -> axis "
        f"rename, found {declared!r}"
    )

    lock = json.loads((FRONTEND / "package-lock.json").read_text(encoding="utf-8"))
    locked = lock["packages"]["node_modules/lucide-react"]["version"]
    assert _version_tuple(locked) >= (1, 74, 0), (
        f"package-lock.json must resolve lucide-react to >= 1.74.0, found {locked!r}"
    )

    removed_icons = re.compile(r"\b(Centerline|Github)\b")
    offenders = []

    for source in (FRONTEND / "src").rglob("*.tsx"):
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if removed_icons.search(line):
                offenders.append(f"{source.relative_to(ROOT)}:{line_number}")

    assert not offenders, (
        "lucide-react 1.74 no longer exports Centerline or Github; use Axis or "
        "remove the unused import: " + ", ".join(offenders)
    )
