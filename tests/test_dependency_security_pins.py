"""Guards the security-pinned versions of the frontend dependencies.

These pins address CVEs in the checked-in manifests (frontend/package.json and
frontend/package-lock.json):

* GHSA-587h-7wmj-h5pm -> vite pinned to the fixed 8.3.3 release.
* GHSA-4wf5-vphf-c2xc -> terser (optional peer of vite) pinned to 4.8.1.

If a future dependency bump regresses any of these pins, this test fails.
"""

import json
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


def test_frontend_vite_and_terser_security_pins():
    pkg = json.loads((FRONTEND / "package.json").read_text())
    lock = json.loads((FRONTEND / "package-lock.json").read_text())

    # GHSA-587h-7wmj-h5pm: vite must be pinned to the fixed release.
    assert pkg["devDependencies"]["vite"] == "8.3.3"
    assert lock["packages"]["node_modules/vite"]["version"] == "8.3.3"

    # GHSA-4wf5-vphf-c2xc: terser (vite optional peer) pinned to the fixed release.
    peers = lock["packages"]["node_modules/vite"]["peerDependencies"]
    assert peers["terser"] == "4.8.1"
