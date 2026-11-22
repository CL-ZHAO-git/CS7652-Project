"""Fetch CRSP data from WRDS and build the forecasting panel.

Run from the repository root; the commands are the same on macOS, Linux, and Windows:

    python scripts/crsp_data.py fetch configs/datasets/crsp_lc30_2017_2019.toml
    python scripts/crsp_data.py build configs/datasets/crsp_lc30_2017_2019.toml

fetch needs WRDS access and the wrds package (python -m pip install -e ".[wrds]").
The first connection asks for your WRDS password and may send a Duo push; wrds can
save the password to your own pgpass file, outside this repository. build works
offline with the standard library. All outputs go under data/, which git ignores.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecast_agents.datasets.crsp import build_panel, fetch_raw, load_spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="download a raw CRSP extract from WRDS")
    fetch.add_argument("spec", type=Path, help="dataset spec TOML, e.g. configs/datasets/<name>.toml")
    fetch.add_argument("--wrds-username", default=os.environ.get("WRDS_USERNAME"),
                       help="WRDS username (default: $WRDS_USERNAME, otherwise WRDS prompts)")
    fetch.add_argument("--force", action="store_true", help="replace an existing raw extract")
    build = commands.add_parser("build", help="build the panel JSON from a raw extract (offline)")
    build.add_argument("spec", type=Path, help="dataset spec TOML, e.g. configs/datasets/<name>.toml")
    build.add_argument("--force", action="store_true", help="replace existing panel files")
    args = parser.parse_args(argv)
    try:
        spec = load_spec(args.spec)
        if args.command == "build":
            summary = build_panel(spec, force=args.force)
        else:
            try:
                import wrds
            except ImportError:
                print('The fetch step needs the wrds package: python -m pip install -e ".[wrds]"',
                      file=sys.stderr)
                return 2
            options = {"wrds_username": args.wrds_username} if args.wrds_username else {}
            db = wrds.Connection(**options)
            try:
                summary = fetch_raw(spec, db, force=args.force)
            finally:
                db.close()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
