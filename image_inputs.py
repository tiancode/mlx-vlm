"""Bounded local image uploads; no remote URLs or server filesystem paths."""
import base64
from io import BytesIO
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_EDIT_BODY = 64 * 1024**2
MAX_IMAGE_BYTES = 10 * 1024**2
MAX_IMAGES = 10
MAX_IMAGE_PIXELS = 16_000_000
MAX_TOTAL_PIXELS = 40_000_000


class InvalidEditPrompt(ValueError):
    pass


def decode_base64_image(value):
    if not isinstance(value, str):
        raise ValueError("Each image must be a base64 string or image data URL")
    if value.startswith("data:"):
        header, sep, value = value.partition(",")
        if not sep or header not in ("data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"):
            raise ValueError("Only PNG, JPEG and WebP base64 data URLs are supported")
    if len(value) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
        raise ValueError("Each image must be at most 10 MiB")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError:
        raise ValueError("Invalid image base64") from None


def validate_images(images):
    if not 1 <= len(images) <= MAX_IMAGES:
        raise ValueError("Provide 1 to 10 images in upload order")
    total = 0
    for data in images:
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Each image must be nonempty and at most 10 MiB")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as img:
                    if img.format not in ("PNG", "JPEG", "WEBP") or getattr(img, "n_frames", 1) != 1:
                        raise ValueError("Only still PNG, JPEG and WebP images are supported")
                    pixels = img.width * img.height
                    total += pixels
                    if pixels > MAX_IMAGE_PIXELS or total > MAX_TOTAL_PIXELS:
                        raise ValueError("Images exceed 16 megapixels each or 40 megapixels total")
                    if max(img.size) / min(img.size) > 16:
                        raise ValueError("Image aspect ratio must be between 1:16 and 16:1")
                    img.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise ValueError("Invalid, damaged or oversized image") from None
    return images


def prepare_reference(data, resolution):
    with Image.open(BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
    ratio = image.width / image.height
    width = max(32, round((resolution**2 * ratio)**0.5 / 32) * 32)
    height = max(32, round((resolution**2 / ratio)**0.5 / 32) * 32)
    return image.resize((width, height), Image.Resampling.LANCZOS)
