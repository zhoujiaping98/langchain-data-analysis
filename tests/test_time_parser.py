from app.time_parser import infer_time_range


def test_infer_last_week():
    tr = infer_time_range("上周GMV")
    assert tr is not None
    assert tr.label == "last_week"
