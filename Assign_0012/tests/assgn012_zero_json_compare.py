"""Compare DeepSpeed ZeRO example JSON stages to lab W=32 accounting.

```text
python tests/assgn012_zero_json_compare.py
```

Exit: 0 PASS, 1 FAIL, 2 crash.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.deepspeed_json_compare import run_compare


def main() -> int:
    print(f"Assign_0012 DeepSpeed JSON compare (root={ROOT})")
    try:
        result = run_compare(ROOT)
    except Exception:
        traceback.print_exc()
        return 2

    for r in result["rows"]:
        print(
            f"  {r.stage_name}: verdict={r.verdict}  "
            f"implied_gib={r.implied_gib:.2f}  lab_gib={r.lab_gib}  "
            f"delta={r.delta_gib}  ok_detail={r.notes}"
        )
    print(f"\nReport: {result.get('report_path')}")
    print(f"Overall: ok={result['ok']}  ({'PASS' if result['ok'] else 'FAIL'})")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
