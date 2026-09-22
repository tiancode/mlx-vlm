from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from image_inputs import prepare_reference, validate_images


class ReferenceInputTests(unittest.TestCase):
    def image_bytes(self, fmt, *, size=(80, 40), **kwargs):
        buf = BytesIO()
        Image.new("RGB", size, "red").save(buf, format=fmt, **kwargs)
        return buf.getvalue()

    def test_supported_formats_and_exif_orientation(self):
        images = [self.image_bytes(fmt) for fmt in ("PNG", "JPEG", "WEBP")]
        self.assertEqual(validate_images(images), images)
        exif = Image.Exif()
        exif[274] = 6
        prepared = prepare_reference(self.image_bytes("JPEG", exif=exif), 512)
        self.assertEqual(prepared.mode, "RGBA")
        self.assertLess(prepared.width, prepared.height)
        self.assertEqual(prepared.width % 32, 0)
        self.assertEqual(prepared.height % 32, 0)

    def test_total_pixels_and_extreme_aspect_ratio_are_bounded(self):
        data = self.image_bytes("PNG")
        with patch("image_inputs.MAX_TOTAL_PIXELS", 5000):
            with self.assertRaisesRegex(ValueError, "total"):
                validate_images([data, data])
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            validate_images([self.image_bytes("PNG", size=(100, 1))])

    def test_alpha_is_preserved_for_vae_input(self):
        buf = BytesIO()
        Image.new("RGBA", (32, 32), (255, 0, 0, 128)).save(buf, format="PNG")
        image = prepare_reference(buf.getvalue(), 512)
        self.assertEqual(image.getpixel((100, 100))[3], 128)


if __name__ == "__main__":
    unittest.main(verbosity=2)
