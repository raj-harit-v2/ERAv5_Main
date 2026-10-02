"""Print the next command when an Assgn_014 gate fails. Does not retrain full budgets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.versioning import resolve_reports_for_eval
from tests.self_evaluate import evaluate

REPORTS_ROOT = ROOT / "reports"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=str, default=None)
    args = parser.parse_args(argv)
    reports = resolve_reports_for_eval(REPORTS_ROOT, reports_arg=args.reports)
    gate = evaluate(smoke_ok=True, reports=reports)
    print("ASSGN014 self_correct - next commands (not auto-run):")
    print(f"reports={reports}")
    if gate["all_ok"]:
        print("all_ok=True  nothing to correct")
        return 0
    if not gate.get("n_ok"):
        print(
            f"N_dense={gate.get('n_dense')} outside 18e6-22e6. "
            "Edit LMConfig in src/llm/tiny_lm.py (n_layer / n_embd) and re-run smoke."
        )
    if not gate.get("moe_bigger"):
        print(
            "MoE total params must exceed dense. Increase ASSGN014_N_EXPERTS "
            "or check upcycle created MoE experts."
        )
        print("python tests/assgn014_moe_lab.py --smoke --tokens-dense 4096 --tokens-moe 4096")
    if not gate.get("token_ok"):
        print("python tests/assgn014_moe_lab.py --smoke --tokens-dense 4096 --tokens-moe 4096")
        print("For full: python tests/assgn014_moe_lab.py --full")
    if not gate.get("loss_ok"):
        print(
            "MoE final_loss did not drop below loss_at_convert. "
            "Re-run with more MoE tokens or check upcycle/bias:"
        )
        print("python tests/assgn014_moe_lab.py --smoke --tokens-dense 8192 --tokens-moe 16384")
    if not gate.get("balance_ok"):
        print(
            f"Routing unhealthy: max_vio={gate.get('max_vio')} dead={gate.get('dead_experts')}. "
            "Confirm bias_gamma>0 and CE-aux OFF. Re-run smoke."
        )
    if not gate.get("png_ok"):
        print("python tests/assgn014_moe_lab.py --smoke --tokens-dense 4096 --tokens-moe 4096")
        print("  (regenerates reports/vNNN_*/assgn014_*.png)")
    if not gate.get("aux_off"):
        print("Set use_ce_aux=False (graded path forbids CE aux load loss).")
    if not gate.get("router_fp32"):
        print("Router must be float32. Check MoELayer.router dtype after .to(device).")
    if gate.get("smoke") and not gate["all_ok"]:
        print("Colab disconnect: set ASSGN014_RUN_VERSION=<id> and continue the same folder.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
