"""Images attached to messages: checked, shrunk, stripped, and only from the request."""
from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

import images


def png(w=400, h=300, mode="RGB", fmt="PNG", **save):
    out = io.BytesIO()
    Image.new(mode, (w, h), "red" if mode != "RGBA" else (255, 0, 0, 0)).save(out, fmt, **save)
    return out.getvalue()


def url(raw, kind="png"):
    return f"data:image/{kind};base64," + base64.b64encode(raw).decode()


def back(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def test_large_images_are_shrunk_and_small_ones_kept():
    assert back(images.prepare(png(4000, 3000))).size == (1600, 1200)
    assert back(images.prepare(png(300, 200))).size == (300, 200)


def test_metadata_is_dropped_and_photos_stay_upright():
    exif = Image.Exif()
    exif[0x0112] = 6                      # rotated: phone held upright
    exif[0x010F] = "PhoneMaker"
    raw = png(400, 300, fmt="JPEG", exif=exif.tobytes())
    out = back(images.prepare(raw))
    assert out.size == (300, 400) and not out.getexif()


@pytest.mark.parametrize("mode", ["RGBA", "P", "L", "LA", "CMYK"])
def test_other_colour_modes(mode):
    raw = png(50, 40, mode=mode, fmt="PNG" if mode != "CMYK" else "JPEG")
    assert back(images.prepare(raw)).mode == "RGB"


def test_animations_use_the_first_frame():
    frames = [Image.new("RGB", (40, 40), c) for c in ("red", "blue")]
    out = io.BytesIO()
    frames[0].save(out, "GIF", save_all=True, append_images=frames[1:])
    assert back(images.prepare(out.getvalue())).size == (40, 40)


@pytest.mark.parametrize("raw", [b"", b"not an image", png()[:100], b"\x89PNG\r\n\x1a\n" + b"\x00" * 50])
def test_unreadable(raw):
    with pytest.raises(images.ImageError, match="could not be read"):
        images.prepare(raw)


def test_size_and_pixel_limits():
    with pytest.raises(images.ImageError, match="too large"):
        images.prepare(b"x" * (images.MAX_BYTES + 1))
    with pytest.raises(images.ImageError):
        images.prepare(png(8000, 6000, mode="L"))  # 48M pixels


@pytest.mark.parametrize("bad", [
    "https://example.com/cat.png", "http://169.254.169.254/x.png", "file:///etc/passwd",
    "data:text/html;base64,PGgxPg==", "data:image/svg+xml;base64,PHN2Zz4=", "data:image/png,raw", "", None, 5,
])
def test_only_attached_images_never_addresses(bad):
    with pytest.raises(images.ImageError):
        images.decode_data_url(bad)


def test_too_many():
    with pytest.raises(images.ImageError, match="At most"):
        images.prepare_all([url(png())] * (images.MAX_IMAGES + 1))


def test_content_parts():
    parts = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "data:x"}},
             {"type": "image_url", "image_url": "data:y"}, "junk", {"type": "other"}]
    assert images.from_content(parts) == ["data:x", "data:y"]
    assert images.from_content("plain") == []


def test_attach_goes_on_the_last_user_message_with_the_rules():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"}, {"role": "user", "content": ""}]
    out = images.attach(msgs, ["IMG"])
    assert out[3] == {"role": "user", "content": "Describe this image.", "images": ["IMG"]}
    assert "isn't shown" in out[0]["content"] and "images" not in out[1] and msgs[3] == {"role": "user", "content": ""}
    assert images.attach(msgs, []) is msgs
    assert images.attach([{"role": "user", "content": "q"}], ["I"])[0]["role"] == "system"
