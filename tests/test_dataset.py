from pandas.testing import assert_frame_equal

from data.generate_dataset import build
from src.common import LABELS


def test_grouped_dataset():
    data = build()
    assert_frame_equal(data, build())
    assert len(data) == data.text.nunique() == data.ticket_id.nunique() == 2000
    assert data.family.nunique() == 80
    assert data.groupby("family").split.nunique().eq(1).all()
    assert set(data.label) == set(LABELS)
    assert data.label.value_counts().eq(250).all()
    assert data.groupby("split").size().to_dict() == {"train": 1400, "validation": 300, "test": 300}
    assert data.groupby(["split", "label"]).size().min() == 25


def test_public_mapping_filters_conflicts_and_duplicates():
    import pandas as pd

    from data.prepare_data import map_tickets

    source = pd.DataFrame([
        ("get_refund", "Please return my money"),
        ("track_refund", " Please return my money "),
        ("get_refund", "Update please"),
        ("get_invoice", " update  please "),
        ("switch_account", "other"),
    ], columns=["intent", "instruction"])
    result = map_tickets(source)
    assert result.label.tolist() == ["Refund"]
    assert result.text.tolist() == ["Please return my money"]
