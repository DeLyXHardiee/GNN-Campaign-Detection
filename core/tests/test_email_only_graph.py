"""Tests for email-only graph config and IR projection."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch_geometric  # noqa: F401
except ImportError:  # pragma: no cover
    torch_geometric = None  # type: ignore

from graph.email_only_projection import project_ir_to_email_only, DEFAULT_IR_TO_RELATION
from graph.graph_schema import DEFAULT_SCHEMA


class TestEmailOnlySettings(unittest.TestCase):
    def test_graph_build_settings_email_only_mode(self) -> None:
        import shutil
        import tempfile

        from config.pipeline_config import graph_build_settings_from_pipeline

        d = Path(tempfile.mkdtemp())
        try:
            misp = d / "inc.json"
            misp.write_text("[]", encoding="utf-8")
            outd = d / "gout"
            outd.mkdir()
            cfg = {
                "graph": {
                    "misp_json_path": str(misp.resolve()),
                    "output_dir": str(outd),
                    "mode": "email_only",
                    "email_only": {
                        "weights": {"has_url": 0.5},
                        "min_emails_per_infra": 3,
                    },
                },
                "datasets": {},
            }
            s = graph_build_settings_from_pipeline(cfg, project_root=d)
            self.assertEqual(s.mode, "email_only")
            self.assertIsNotNone(s.email_only)
            assert s.email_only is not None
            self.assertEqual(s.email_only.relation_weights["has_url"], 0.5)
            self.assertEqual(s.email_only.min_emails_per_infra, 3)
            base, _ = os.path.splitext(os.path.basename(s.misp_json_path))
            expected = os.path.join(s.output_dir, f"{base}_email_only.pt")
            self.assertTrue(expected.endswith("inc_email_only.pt"))
        finally:
            shutil.rmtree(d, ignore_errors=True)


def _minimal_ir(
    n_email: int,
    has_url: tuple[list[int], list[int]] | None,
) -> SimpleNamespace:
    xs = [[0.0] * 4 for _ in range(n_email)]
    email = SimpleNamespace(x=xs, index_to_meta=[{"index": i} for i in range(n_email)])
    edges: dict = {}
    if has_url is not None:
        edges["has_url"] = has_url
    return SimpleNamespace(nodes={"email": email}, edges=edges, email_attrs={})


@unittest.skipIf(torch_geometric is None, "torch_geometric not installed")
class TestEmailOnlyProjection(unittest.TestCase):
    def test_project_ir_shared_url_pair(self) -> None:
        ir = _minimal_ir(2, ([0, 1], [0, 0]))
        data, meta = project_ir_to_email_only(
            ir,
            DEFAULT_SCHEMA,
            enabled_ir_edges={"has_url"},
            min_emails_per_infra=2,
        )
        self.assertNotIn("x", data["email"])
        self.assertIn(("email", "aggregated", "email"), data.edge_types)
        ei = data["email", "aggregated", "email"].edge_index
        self.assertEqual(ei.shape[1], 2)
        pairs = set((ei[0, i].item(), ei[1, i].item()) for i in range(2))
        self.assertEqual(pairs, {(0, 1), (1, 0)})
        self.assertIn("email_only", meta)
        w = data["email", "aggregated", "email"].edge_attr
        self.assertIsNotNone(w)
        self.assertEqual(float(w[0, 0].item()), 1.0)
        self.assertEqual(float(w[1, 0].item()), 1.0)

    def test_sum_two_distinct_infras_doubles_weight(self) -> None:
        # emails 0,1 both touch url0 and url1 -> two contributions, default weight 1 -> sum=2
        ir = _minimal_ir(2, ([0, 1, 0, 1], [0, 0, 1, 1]))
        data, _ = project_ir_to_email_only(
            ir,
            DEFAULT_SCHEMA,
            enabled_ir_edges={"has_url"},
            pair_weight_aggregation="sum",
        )
        w = data["email", "aggregated", "email"].edge_attr
        self.assertIsNotNone(w)
        self.assertEqual(float(w[0, 0].item()), 2.0)

    def test_project_ir_relation_weight(self) -> None:
        ir = _minimal_ir(2, ([0, 1], [0, 0]))
        data, _ = project_ir_to_email_only(
            ir,
            DEFAULT_SCHEMA,
            enabled_ir_edges={"has_url"},
            relation_weights={"has_url": 3.0},
        )
        w = data["email", "aggregated", "email"].edge_attr
        self.assertIsNotNone(w)
        self.assertEqual(float(w[0, 0].item()), 3.0)

    def test_min_emails_per_infra_suppresses_edge(self) -> None:
        ir = _minimal_ir(2, ([0], [0]))
        data, _ = project_ir_to_email_only(
            ir,
            DEFAULT_SCHEMA,
            enabled_ir_edges={"has_url"},
            min_emails_per_infra=2,
        )
        if len(data.edge_types) > 0:
            ei = data["email", "aggregated", "email"].edge_index
            self.assertEqual(ei.shape[1], 0)

    def test_default_ir_to_relation_includes_senders(self) -> None:
        self.assertIn("sender_from_domain", DEFAULT_IR_TO_RELATION)
        self.assertIn("receiver_from_domain", DEFAULT_IR_TO_RELATION)


if __name__ == "__main__":
    unittest.main()
