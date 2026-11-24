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


def test_support_router_catalog_and_family_holdout():
    from data.generate_destinations import REGIONS, TAXONOMY, build_destinations
    from data.generate_tickets import build_tickets

    catalog = build_destinations()
    by_id = {row["destination_id"]: row for row in catalog}
    assert len(catalog) == len(by_id) == 2000
    assert all(row["subcategory"] in TAXONOMY[row["department"]] for row in catalog)
    assert {row["region"] for row in catalog} == set(REGIONS)
    tickets = build_tickets()
    assert tickets.groupby("split").size().to_dict() == {"train": 8000, "validation": 2000, "test": 2000}
    assert tickets.ticket_text.nunique() == len(tickets)
    assert tickets.groupby("template_family").split.nunique().eq(1).all()
    assert tickets.groupby("split").department.nunique().eq(8).all()
    assert set(tickets.destination_id) == set(by_id)
    assert all(by_id[row.destination_id]["region"] == row.region for row in tickets.itertuples())


def test_router_csv_preserves_north_america(tmp_path, monkeypatch):
    from data import prepare_dataset

    monkeypatch.setattr(prepare_dataset, "DATA", tmp_path)
    (tmp_path / "train.csv").write_text("region\nNA\n")
    assert prepare_dataset.read_tickets("train").region.tolist() == ["NA"]
