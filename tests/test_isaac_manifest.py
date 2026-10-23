"""Unit tests for Isaac Sim canonical scene manifest loader and validator."""
import json
from pathlib import Path
import tempfile
import unittest

from simulation.isaac_manifest import (
    DEFAULT_MANIFEST_PATH,
    MANIFEST_SCHEMA_VERSION,
    IsaacSceneManifest,
    load_manifest,
)


class TestIsaacManifest(unittest.TestCase):
    def test_load_default_manifest(self):
        manifest = load_manifest()
        self.assertIsInstance(manifest, IsaacSceneManifest)
        self.assertEqual(manifest.schema_version, MANIFEST_SCHEMA_VERSION)
        self.assertEqual(manifest.default_stage_file, "scenes/fabric_feed_cutting_world.usda")
        self.assertEqual(manifest.machine_root, "/World/AirTable")
        self.assertEqual(manifest.preview_namespace, "/World/JetsonPreview")
        self.assertIn("/World/FeedHead", manifest.legacy_aliases)
        self.assertIn("verified_centerline", manifest.preview_prims)
        self.assertEqual(
            manifest.preview_prims["verified_centerline"],
            "/World/JetsonPreview/VerifiedCenterline",
        )
        self.assertFalse(manifest.execution_authorized)

    def test_manifest_immutability(self):
        manifest = load_manifest()
        with self.assertRaises((AttributeError, TypeError)):
            manifest.schema_version = "2.0.0"
        with self.assertRaises((AttributeError, TypeError)):
            manifest.target_scene["new_key"] = "val"

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_manifest(Path("/nonexistent/manifest.json"))

    def test_invalid_schema_version_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps({"schema_version": "9.9.9"}))
            temp_path = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                load_manifest(temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_execution_authorized_true_is_blocked(self):
        with open(DEFAULT_MANIFEST_PATH, "r") as f:
            data = json.load(f)
        data["safety_enforcement"]["execution_authorized"] = True
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = Path(f.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_manifest(temp_path)
            self.assertIn("execution_authorized must be False", str(ctx.exception))
        finally:
            if temp_path.exists():
                temp_path.unlink()


if __name__ == "__main__":
    unittest.main()
