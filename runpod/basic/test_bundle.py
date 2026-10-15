"""CPU-only checks of graph integrity, fair comparison and download failure handling."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bootstrap
import build_workflows as workflows

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / "workflows" / name).read_text(encoding="utf-8"))


class BundleTests(unittest.TestCase):
    def test_exact_upstream_and_reproducible_presets(self):
        before = {p.name: p.read_bytes() for p in (ROOT / "workflows").glob("*.json")}
        workflows.build()
        self.assertEqual(before, {p.name: p.read_bytes() for p in (ROOT / "workflows").glob("*.json")})

    def test_all_graph_links_and_only_locked_weights(self):
        manifest = json.loads((ROOT / "models.lock.json").read_text())
        available = {Path(x["file"]).name for x in manifest["models"]}
        presets = list((ROOT / "workflows").glob("*.json"))
        self.assertEqual(len(presets), 5)
        for p in presets:
            graph = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(graph["links"], workflows.rebuild(copy.deepcopy(graph))["links"])
            for n in graph["nodes"]:
                self.assertEqual(n["mode"], 0)
                for value in n.get("widgets_values", []):
                    if isinstance(value, str) and value.endswith(".safetensors"):
                        self.assertIn(value, available, (p.name, value))
                        self.assertNotIn("turbo", value.lower())

    def test_reference_comparison_changes_checkpoint_only(self):
        original = load("01_REF_Oficial_BF16.json")
        candidate = load("05_REF_Singularity_INT8_Comparacao.json")
        self.assertEqual(workflows.node(original, "PrimitiveInt")["widgets_values"][0], 20)
        self.assertEqual(workflows.node(candidate, "RandomNoise")["widgets_values"], [42, "fixed"])
        self.assertFalse([n for n in candidate["nodes"] if n["type"] == "LoraLoaderModelOnly"])
        workflows.node(candidate, "UNETLoader")["widgets_values"][0] = workflows.REF
        for graph in (original, candidate):
            graph.pop("id")
            graph.pop("extra")
            workflows.node(graph, "SaveVideo")["widgets_values"][0] = "comparison"
        self.assertEqual(original, candidate)

    def test_action_model_reaches_scheduler_and_guider(self):
        for name in ("03_REF_Acao_BF16.json", "04_FirstLast_Acao_BF16.json"):
            g = load(name)
            loras = [n for n in g["nodes"] if n["type"] == "LoraLoaderModelOnly"]
            self.assertEqual([n["widgets_values"] for n in loras],
                [["H3_Combat_V2.safetensors", 0.8], ["Motion_Repair_V2.safetensors", 0.6]])
            final = loras[-1]["id"]
            for kind in ("BasicScheduler", "BasicGuider"):
                target = workflows.node(g, kind)
                incoming = next(x for x in g["links"] if x[3] == target["id"] and x[4] == 0)
                self.assertEqual(incoming[1], final)
            self.assertEqual(workflows.node(g, "PrimitiveInt")["widgets_values"][0], 30)

    def test_first_and_last_frame_are_both_connected(self):
        for name in ("02_FirstLast_Oficial_BF16.json", "04_FirstLast_Acao_BF16.json"):
            g = load(name)
            target = workflows.node(g, "MiniMaxH3ImageToVideo")
            connected = {x[4] for x in g["links"] if x[3] == target["id"]}
            self.assertIn(2, connected)
            self.assertIn(3, connected)

    def test_zero_volumes_and_eight_explicit_files(self):
        template = json.loads((ROOT / "runpod-template.v2.proposed.json").read_text())
        manifest = json.loads((ROOT / "models.lock.json").read_text())
        self.assertEqual(template["disk"], 200)
        self.assertEqual(template["mounts"], {})
        self.assertEqual(len(manifest["models"]), 8)
        self.assertEqual(sum(x["size_bytes"] for x in manifest["models"]), manifest["total_size_bytes"])
        self.assertEqual(manifest["total_size_bytes"], 172085535086)
        for item in manifest["models"]:
            self.assertRegex(item["revision"], r"^[a-f0-9]{40}$")
            self.assertRegex(item["sha256_published"], r"^[a-f0-9]{64}$")
            path = Path(item["destination"])
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)

    def test_verified_download_moves_without_second_copy(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".test-download-") as directory:
            root = Path(directory)
            payload = b"small test payload, not learned weights"
            item = {"repository": "test/model", "revision": "a" * 40, "file": "model.safetensors",
                "destination": "models/diffusion_models/model.safetensors", "size_bytes": len(payload),
                "sha256_published": hashlib.sha256(payload).hexdigest()}
            staged = root / "stage/model.safetensors"
            staged.parent.mkdir()
            staged.write_bytes(payload)
            result = bootstrap.download_one(item, root, None, lambda **kwargs: str(staged))
            self.assertEqual(result.read_bytes(), payload)
            self.assertFalse(staged.exists())
            self.assertEqual(bootstrap.download_one(item, root, None, None), result)

    def test_download_hash_failure_is_fatal(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".test-corrupt-") as directory:
            root = Path(directory)
            staged = root / "corrupt.safetensors"
            staged.write_bytes(b"wrong")
            item = {"repository": "test/model", "revision": "a" * 40, "file": staged.name,
                "destination": "models/diffusion_models/model.safetensors", "size_bytes": 5,
                "sha256_published": "0" * 64}
            with patch.object(bootstrap.time, "sleep"):
                with self.assertRaises(RuntimeError):
                    bootstrap.download_one(item, root, None, lambda **kwargs: str(staged))
            self.assertFalse((root / item["destination"]).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
