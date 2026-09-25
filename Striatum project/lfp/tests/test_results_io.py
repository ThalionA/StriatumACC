"""The one CSV writer every driver uses."""

import csv

from striatum_lfp import results_io


def test_write_rows_takes_the_union_of_keys_in_first_seen_order(tmp_path):
    path = tmp_path / "t.csv"
    results_io.write_rows([{"a": 1, "b": 2}, {"a": 3, "c": 4}], path)
    with path.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert reader.fieldnames == ["a", "b", "c"]
    assert rows[1] == {"a": "3", "b": "", "c": "4"}


def test_write_rows_writes_nothing_for_no_rows(tmp_path):
    path = tmp_path / "t.csv"
    results_io.write_rows([], path)
    assert not path.exists()
