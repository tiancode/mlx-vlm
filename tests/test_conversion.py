"""Conversion integrity checks without MLX or model weights."""

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_checkpoint import read_index
from finalize_mxfp8 import check_weights, finalize, MXFP8


def write_tensor(path, name="layer.scales"):
    header = json.dumps({name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))


class ConversionIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_rebuild_index_from_actual_headers(self):
        write_tensor(self.root / "model-00001-of-00001.safetensors", "layer.weight_scale_inv")
        index = read_index(self.root)
        self.assertEqual(index["metadata"]["total_size"], 4)
        self.assertEqual(index["weight_map"], {"layer.weight_scale_inv": "model-00001-of-00001.safetensors"})

    def test_missing_shard_rejected(self):
        write_tensor(self.root / "model-00001-of-00002.safetensors")
        with self.assertRaisesRegex(ValueError, "分片不齐全"):
            read_index(self.root)

    def test_truncated_shard_rejected(self):
        path = self.root / "model-00001-of-00001.safetensors"
        write_tensor(path)
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "分片大小不匹配"):
            read_index(self.root)

    def test_duplicate_tensor_rejected(self):
        for i in (1, 2):
            write_tensor(self.root / f"model-{i:05d}-of-00002.safetensors")
        with self.assertRaisesRegex(ValueError, "张量重复"):
            read_index(self.root)

    def test_mtp_without_index_and_finetuned_template(self):
        source, target = self.root / "source", self.root / "target"
        source.mkdir()
        target.mkdir()
        (source / "chat_template.jinja").write_text("finetuned template")
        (target / "config.json").write_text(json.dumps({"model_type": "glm5_next_mtp"}))
        write_tensor(target / "model.safetensors")
        finalize(source, target)
        self.assertEqual(check_weights(target), (1, 1))
        self.assertEqual((target / "chat_template.jinja").read_text(), "finetuned template")
        config = json.loads((target / "config.json").read_text())
        self.assertEqual(config["model_type"], "glm5_next_mtp")
        self.assertEqual(config["quantization"], MXFP8)

    def test_missing_output_shard_rejected(self):
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"layer.scales": "missing.safetensors"}}))
        with self.assertRaisesRegex(SystemExit, "缺少权重分片"):
            check_weights(self.root)

    def test_unconverted_fp8_cannot_be_finalized(self):
        write_tensor(self.root / "model.safetensors", "layer.weight_scale_inv")
        with self.assertRaises(SystemExit):
            check_weights(self.root)

    def test_truncated_output_does_not_change_config(self):
        path = self.root / "model.safetensors"
        write_tensor(path)
        path.write_bytes(path.read_bytes()[:-1])
        config = self.root / "config.json"
        config.write_text('{}')
        with self.assertRaisesRegex(ValueError, "分片大小不匹配"):
            finalize(self.root, self.root)
        self.assertEqual(config.read_text(), '{}')

    def test_index_cannot_invent_scales(self):
        name = "model-00001-of-00001.safetensors"
        write_tensor(self.root / name, "layer.weight_scale_inv")
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"layer.scales": name}}))
        with self.assertRaisesRegex(ValueError, "索引与实际分片不匹配"):
            check_weights(self.root)

    def test_packed_mxfp8_dtypes(self):
        header = json.dumps({
            "layer.weight": {"dtype": "U32", "shape": [8], "data_offsets": [0, 32]},
            "layer.scales": {"dtype": "U8", "shape": [1], "data_offsets": [32, 33]},
        }).encode()
        (self.root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(33))
        self.assertEqual(check_weights(self.root), (2, 1))

    def test_short_header_rejected(self):
        (self.root / "model.safetensors").write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, "截断的 safetensors header"):
            check_weights(self.root)


if __name__ == "__main__":
    unittest.main()
