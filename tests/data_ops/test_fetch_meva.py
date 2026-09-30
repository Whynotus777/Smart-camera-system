from collections import Counter

from data_ops.fetch import meva


def test_meva_camera_parsing():
    key = "drops-123-r13/2018-03-05/09/2018-03-05.09-49-41.09-50-01.school.G339.r13.avi"
    assert meva.camera_of(key) == ("2018-03-05.09-49-41.09-50-01.school.G339", "school.G339")
    assert meva.camera_of("drops-123-r13/readme.txt") is None


def test_meva_etag():
    assert meva._etag_checksum('"5d41402abc4b2a76b9719d911017c592"') == "md5:5d41402abc4b2a76b9719d911017c592"
    assert meva._etag_checksum('"abc-12"') == "s3etag:abc-12"  # multipart: checked part-wise


def _obj(clip, size):
    return {"Key": f"drops-123-r13/x/{clip}.r13.avi", "Size": size, "ETag": '"0"'}


def test_meva_select_indoor_annotated_first_and_capped():
    objs = [
        _obj("2018-03-05.10-00-00.10-05-00.school.G330", 40),  # indoor, not annotated
        _obj("2018-03-05.09-00-00.09-05-00.school.G421", 40),  # indoor, annotated
        _obj("2018-03-05.08-00-00.08-05-00.school.G339", 10),  # outdoor
        _obj("2018-03-05.07-00-00.07-05-00.bus.G331", 30),  # indoor, annotated
        _obj("2018-03-05.06-00-00.06-05-00.admin.G326", 50),  # indoor, not annotated, too big once full
    ]
    ann = {"2018-03-05.09-00-00.09-05-00.school.G421": Counter(),
           "2018-03-05.07-00-00.07-05-00.bus.G331": Counter()}
    picked = [c for _, c, _ in meva.select(objs, ann, max_bytes=110)]
    assert picked == ["2018-03-05.07-00-00.07-05-00.bus.G331", "2018-03-05.09-00-00.09-05-00.school.G421",
                      "2018-03-05.10-00-00.10-05-00.school.G330"]


def test_meva_annotation_index(tmp_path):
    base = tmp_path / "annotation" / "DIVA-phase-2" / "MEVA"
    a = base / "kitware" / "d" / "c1.activities.yml"
    b = base / "kitware-meva-training" / "d" / "c1.activities.yml"
    c = base / "kitware-meva-training" / "d" / "c2.activities.yml"
    for f in (a, b, c):
        f.parent.mkdir(parents=True, exist_ok=True)
    a.write_text("- {'act': {'act2': {'person_picks_up_object': 1.0}}}\n" * 2
                 + "- {'act': {'act2': {'person_steals_object': 1.0}}}\n")
    b.write_text("- {'act': {'act2': {'person_picks_up_object': 1.0}}}\n" * 9)  # duplicate clip: ignored
    c.write_text("- {'act': {'act2': {'person_transfers_object': 1.0}}}\n")
    idx = meva.annotation_index(tmp_path)
    assert idx["c1"]["person_picks_up_object"] == 2 and idx["c1"]["person_steals_object"] == 1
    assert idx["c2"]["person_transfers_object"] == 1
