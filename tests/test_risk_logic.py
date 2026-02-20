import ast
import math
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


AA_PATH = Path(__file__).resolve().parents[1] / "aa.py"


def _load_risk_symbols():
    source = AA_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(AA_PATH))
    keep_assigns = {"STYLE_RULES"}
    keep_funcs = {
        "_as_float",
        "normalize_style_name",
        "style_rule",
        "hard_roi_limits_by_style",
        "_rr_floor_by_style",
        "_style_hard_tp_cap_roi",
        "apply_hard_roi_caps",
        "validate_trade_plan",
        "evaluate_microstructure_derivatives_gate",
        "_price_from_roi_target",
        "_pct_from_entry",
        "estimate_roi_from_price",
        "_startup_reconcile_style_from_position",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in keep_assigns:
                    body.append(node)
                    break
        elif isinstance(node, ast.FunctionDef) and node.name in keep_funcs:
            body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {
        "Any": Any,
        "Dict": Dict,
        "Optional": Optional,
        "math": math,
        "np": np,
    }
    exec(compile(module, filename=str(AA_PATH), mode="exec"), ns, ns)
    return ns


class RiskLogicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_risk_symbols()

    def test_scalp_tp_cap_is_configurable(self):
        apply_hard_roi_caps = self.ns["apply_hard_roi_caps"]
        cfg = {
            "hard_cap_scalp_tp_roi": 1.8,
            "hard_cap_scalp_sl_roi": 5.0,
            "scalp_rr_floor": 1.2,
        }
        out = {"tp_pct": 5.0, "sl_pct": 1.0, "leverage": 20}
        res = apply_hard_roi_caps(out, "스캘핑", cfg)
        self.assertAlmostEqual(float(res["tp_pct"]), 1.8, places=8)

    def test_apply_hard_caps_never_shrinks_sl_when_tp_capped(self):
        apply_hard_roi_caps = self.ns["apply_hard_roi_caps"]
        cfg = {
            "hard_cap_day_tp_roi": 3.0,
            "hard_cap_day_sl_roi": 10.0,
            "day_rr_floor": 2.5,
        }
        out = {"tp_pct": 10.0, "sl_pct": 2.0, "leverage": 10}
        res = apply_hard_roi_caps(out, "단타", cfg)
        self.assertAlmostEqual(float(res["tp_pct"]), 3.0, places=8)
        self.assertAlmostEqual(float(res["sl_pct"]), 2.0, places=8)
        self.assertIn("SL 축소 금지", str(res.get("_rr_guard_note", "")))

    def test_validate_trade_plan_rejects_tight_stop(self):
        validate_trade_plan = self.ns["validate_trade_plan"]
        cfg = {
            "min_stop_price_pct": 0.20,
            "min_stop_spread_mult": 1.5,
            "min_stop_spread_floor_pct": 0.03,
            "min_stop_atr_mult": 0.20,
            "min_stop_atr_floor_pct": 0.05,
        }
        plan = validate_trade_plan(
            symbol="BTC/USDT:USDT",
            style="스캘핑",
            decision="buy",
            entry_price=100.0,
            leverage=20.0,
            sl_pct_roi=4.0,
            tp_pct_roi=10.0,
            sl_price_pct=0.30,
            tp_price_pct=1.20,
            orderbook_ctx={"spread_pct": 0.10},
            atr_price_pct=0.50,
            cfg=cfg,
        )
        self.assertFalse(bool(plan["ok"]))
        self.assertEqual(str(plan["reason_code"]), "STOP_TOO_TIGHT")
        self.assertGreater(float(plan["required_stop_price_pct"]), float(plan["sl_price_pct"]))

    def test_validate_trade_plan_accepts_reasonable_distances(self):
        validate_trade_plan = self.ns["validate_trade_plan"]
        cfg = {
            "min_stop_price_pct": 0.20,
            "min_stop_spread_mult": 1.2,
            "min_stop_spread_floor_pct": 0.02,
            "min_stop_atr_mult": 0.10,
            "min_stop_atr_floor_pct": 0.03,
        }
        plan = validate_trade_plan(
            symbol="ETH/USDT:USDT",
            style="단타",
            decision="sell",
            entry_price=2000.0,
            leverage=10.0,
            sl_pct_roi=12.0,
            tp_pct_roi=24.0,
            sl_price_pct=1.20,
            tp_price_pct=2.40,
            orderbook_ctx={"spread_pct": 0.05},
            atr_price_pct=0.40,
            cfg=cfg,
        )
        self.assertTrue(bool(plan["ok"]))
        self.assertEqual(str(plan["reason_code"]), "OK")
        self.assertGreater(float(plan["rr_price"]), 1.0)

    def test_micro_gate_blocks_wide_spread(self):
        micro_gate = self.ns["evaluate_microstructure_derivatives_gate"]
        cfg = {
            "micro_entry_filter_enable": True,
            "micro_max_spread_bps_scalp": 10.0,
            "micro_min_depth_usdt_scalp": 10000.0,
            "micro_block_on_opp_pressure": True,
            "micro_opp_pressure_imbalance": 0.20,
            "micro_funding_filter_enable": False,
            "micro_open_interest_filter_enable": False,
        }
        res = micro_gate(
            symbol="BTC/USDT:USDT",
            decision="buy",
            style="스캘핑",
            orderbook_ctx={
                "available": True,
                "spread_pct": 0.25,  # 25bps
                "depth_notional_usdt": 500000.0,
                "imbalance": 0.10,
                "pressure_side": "buy",
                "pressure_score": 55.0,
            },
            derivatives_ctx={},
            cfg=cfg,
        )
        self.assertFalse(bool(res["ok"]))
        self.assertEqual(str(res["reason_code"]), "SPREAD_TOO_WIDE")

    def test_micro_gate_degrades_when_derivatives_unavailable(self):
        micro_gate = self.ns["evaluate_microstructure_derivatives_gate"]
        cfg = {
            "micro_entry_filter_enable": True,
            "micro_max_spread_bps_scalp": 20.0,
            "micro_min_depth_usdt_scalp": 10000.0,
            "micro_block_on_opp_pressure": True,
            "micro_opp_pressure_imbalance": 0.20,
            "micro_funding_filter_enable": True,
            "micro_funding_block_enable": True,
            "micro_open_interest_filter_enable": True,
            "micro_open_interest_require_confirm": False,
            "micro_open_interest_confirm_min_change_pct": 1.0,
        }
        res = micro_gate(
            symbol="ETH/USDT:USDT",
            decision="sell",
            style="스캘핑",
            orderbook_ctx={
                "available": True,
                "spread_pct": 0.05,
                "depth_notional_usdt": 80000.0,
                "imbalance": -0.12,
                "pressure_side": "sell",
                "pressure_score": 45.0,
            },
            derivatives_ctx={
                "funding": {"supported": False, "available": False, "rate": None},
                "open_interest": {"supported": False, "available": False, "value": None},
                "oi_change_pct": None,
            },
            cfg=cfg,
        )
        self.assertTrue(bool(res["ok"]))
        self.assertEqual(str(res["reason_code"]), "OK_WITH_WARNINGS")
        self.assertTrue(any("FUNDING_UNAVAILABLE" in w for w in (res.get("warnings", []) or [])))
        self.assertTrue(any("OI_UNAVAILABLE" in w for w in (res.get("warnings", []) or [])))

    def test_price_roi_conversion_long_roundtrip(self):
        to_price = self.ns["_price_from_roi_target"]
        to_price_pct = self.ns["_pct_from_entry"]
        estimate_roi = self.ns["estimate_roi_from_price"]
        entry = 100.0
        lev = 20.0
        target_roi = 10.0
        tp = to_price(entry, "buy", target_roi, lev, "tp")
        self.assertIsNotNone(tp)
        price_pct = to_price_pct(entry, "buy", tp, is_tp=True)
        self.assertAlmostEqual(float(price_pct), 0.5, places=8)
        roi = estimate_roi(entry, float(tp), "long", lev)
        self.assertAlmostEqual(float(roi), target_roi, places=8)

    def test_price_roi_conversion_short_roundtrip(self):
        to_price = self.ns["_price_from_roi_target"]
        to_price_pct = self.ns["_pct_from_entry"]
        estimate_roi = self.ns["estimate_roi_from_price"]
        entry = 2500.0
        lev = 10.0
        target_roi = 7.5
        tp = to_price(entry, "sell", target_roi, lev, "tp")
        self.assertIsNotNone(tp)
        price_pct = to_price_pct(entry, "sell", tp, is_tp=True)
        self.assertAlmostEqual(float(price_pct), 0.75, places=8)
        roi = estimate_roi(entry, float(tp), "short", lev)
        self.assertAlmostEqual(float(roi), target_roi, places=8)

    def test_startup_reconcile_style_auto_by_leverage(self):
        fn = self.ns["_startup_reconcile_style_from_position"]
        cfg = {"startup_reconcile_import_style": "auto"}
        self.assertEqual(fn(cfg, {"leverage": 20}), "스캘핑")
        self.assertEqual(fn(cfg, {"leverage": 8}), "단타")
        self.assertEqual(fn(cfg, {"leverage": 3}), "스윙")


if __name__ == "__main__":
    unittest.main()
