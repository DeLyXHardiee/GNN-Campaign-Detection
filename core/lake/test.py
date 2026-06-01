from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_CORE_ROOT = Path(__file__).resolve().parent.parent
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from lake.client import LakeAPIClient


def _load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    _load_env_file(Path(__file__).resolve().parent / ".env")
    _load_env_file(_CORE_ROOT.parent / ".env")

    base_url = os.getenv("LAKE_BASE_URL")
    api_key = os.getenv("LAKE_API_KEY")
    incidents_table = os.getenv("LAKE_INCIDENTS_TABLE", "intellagent.public.incidents")
    parsed_emails_table = os.getenv("LAKE_PARSED_EMAILS_TABLE", "parsed_emails")

    if not base_url or not api_key:
        raise RuntimeError(
            "Missing required env vars. Set LAKE_BASE_URL and LAKE_API_KEY before running this script."
        )

    client = LakeAPIClient(base_url=base_url, api_key=api_key)

    incidents_schema = client.schema(incidents_table)
    parsed_emails_schema = client.schema(parsed_emails_table)

    print(f"Schema for {incidents_table}:")
    print(json.dumps(incidents_schema, indent=2))
    print()
    print(f"Schema for {parsed_emails_table}:")
    print(json.dumps(parsed_emails_schema, indent=2))


if __name__ == "__main__":
    main()