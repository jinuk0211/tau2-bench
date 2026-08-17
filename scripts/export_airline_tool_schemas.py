"""Export the exact airline tool schemas used by the J-Lens HF chat template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau2.domains.airline.environment import get_environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schemas = [tool.openai_schema for tool in get_environment().get_tools()]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"domain": "airline", "tools": schemas}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), "tools": len(schemas)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
