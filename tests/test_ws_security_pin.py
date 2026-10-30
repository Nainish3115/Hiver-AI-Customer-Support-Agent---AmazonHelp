"""Guards the security pin for the transitive "ws" dependency (CVE fix).

GHSA-6663-c963-2gqg affects ws < 8.17.1 (uncontrolled resource consumption /
denial of service). The frontend only depends on ws transitively through vite,
so we pin it with an npm override so every resolution of ws uses a fixed release.
If a future dependency bump regresses this pin, this test fails.
"""

import json
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


def test_frontend_ws_security_pin():
    pkg = json.loads((FRONTEND / "package.json").read_text())
    lock = json.loads((FRONTEND / "package-lock.json").read_text())

    # GHSA-6663-c963-2gqg: ws must be forced to a fixed release (>= 8.17.1).
    pinned = pkg["overrides"]["ws"]
    resolved = lock["packages"]["node_modules/ws"]["version"]

    assert pinned == "8.22.0"
    assert resolved == pinned
