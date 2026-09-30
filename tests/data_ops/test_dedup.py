import shutil
import subprocess

import cv2
import numpy as np
import pytest

from data_ops.dedup import phash
from data_ops.dedup.groups import Grouper, UnionFind, meva_view_group


def _img(seed, size=(360, 640)):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (18, 32, 3), dtype=np.uint8)
    base = cv2.resize(small, size[::-1], interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(base, (0, 0), 3)


def test_phash_robust_to_reencode_resize_but_separates_scenes():
    a = _img(1)
    ok, enc = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 30])
    reencoded = cv2.resize(cv2.imdecode(enc, cv2.IMREAD_COLOR), (320, 180))
    brighter = cv2.convertScaleAbs(a, alpha=1.0, beta=20)
    other = _img(2)
    h = phash.phash(a)
    assert phash.hamming(h, phash.phash(reencoded)) <= 6
    assert phash.hamming(h, phash.phash(brighter)) <= 6
    assert phash.hamming(h, phash.phash(other)) > 16


def test_signature_distance():
    assert phash.signature_distance([0b1010, 0], [0b1010, 0]) == 0
    assert phash.signature_distance([0, 0, 0], [0b1, 0b11, 0b111]) == 2
    assert phash.signature_distance([], [1]) == 64


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_video_signature_matches_across_codecs(tmp_path):
    src, h265, other = tmp_path / "a.mp4", tmp_path / "a_h265.mp4", tmp_path / "b.mp4"
    ff = [shutil.which("ffmpeg"), "-v", "error", "-y"]
    subprocess.run([*ff, "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=4",  # noqa: S603
                    "-c:v", "libx264", str(src)], check=True)
    subprocess.run([*ff, "-i", str(src), "-c:v", "libx265", "-crf", "35", "-vf", "scale=480:270",  # noqa: S603
                    str(h265)], check=True)
    subprocess.run([*ff, "-f", "lavfi", "-i", "mandelbrot=size=640x360:rate=30", "-t", "4",  # noqa: S603
                    "-c:v", "libx264", str(other)], check=True)
    s1, s2, s3 = (phash.video_signature(p) for p in (src, h265, other))
    assert len(s1) == phash.SAMPLES
    assert phash.signature_distance(s1, s2) <= phash.NEAR_DUP
    assert phash.signature_distance(s1, s3) > phash.NEAR_DUP


def test_union_find_and_groups():
    uf = UnionFind()
    uf.union("c", "b")
    uf.union("b", "a")
    assert uf.find("c") == uf.find("a") == "a"

    g = Grouper()
    g.add("meva:x", "meva", "meva:slot1")
    g.add("meva:y", "meva", "meva:slot1")  # same view group, but NOT the same group_id
    g.add("replay:cam01_h264.mp4", "replay")
    g.link("replay:cam01_h264.mp4", "meva:x", "derivative:replay_loop")
    rows = {r["item"]: r for r in g.rows()}
    assert rows["replay:cam01_h264.mp4"]["group_id"] == rows["meva:x"]["group_id"]
    assert rows["meva:y"]["group_id"] != rows["meva:x"]["group_id"]
    assert rows["meva:x"]["view_group"] == rows["meva:y"]["view_group"] == "meva:slot1"
    assert rows["meva:x"]["reasons"] == ["derivative:replay_loop"] and rows["meva:x"]["group_size"] == 2
    g2 = Grouper()  # stable ids: adding unrelated items doesn't change existing group ids
    for item in ("meva:x", "replay:cam01_h264.mp4", "meva:y", "meva:zzz"):
        g2.add(item, "d")
    g2.link("replay:cam01_h264.mp4", "meva:x", "r")
    assert {r["item"]: r["group_id"] for r in g2.rows()}["meva:x"] == rows["meva:x"]["group_id"]


def test_meva_view_group():
    clip = "2018-03-07.17-37-06.17-42-06.school.G339"
    assert meva_view_group(clip) == "meva:2018-03-07.17-35-00.school"
    assert meva_view_group(clip, {clip: "2018-03-07.17-40-00"}) == "meva:2018-03-07.17-40-00.school"
