from __future__ import annotations

import csv
import io
import json
import math

import pandas as pd
import pytest

from preprocessing.patent_detail.pipeline import (
    aggregate_quarterly_metrics,
    build_pair_events,
    canonicalize_records,
    normalize_stock_code,
    parse_ipc_subclasses,
    run_pipeline,
)


def _record(
    application_no: str,
    stkcd: str,
    relation: str = "上市公司本身",
    application_date: str = "1985-01-01",
    ipc: str = "A01B1/00; G06F1/00",
    patent_type: str = "发明申请",
) -> dict[str, str]:
    return {
        "stkcd_raw": stkcd,
        "company_name": f"company-{stkcd}",
        "relation": relation,
        "applicant": f"applicant-{stkcd}",
        "applicant_type": "企业",
        "application_no": application_no,
        "application_date": application_date,
        "publication_date": application_date,
        "grant_date": "",
        "publication_no": "",
        "grant_no": "",
        "patent_type": patent_type,
        "ipc_all_raw": ipc,
        "ipc_main_raw": ipc.split(";")[0],
    }


def test_stock_code_and_ipc_subclass_normalization_are_lossless_for_baseline_fields():
    assert normalize_stock_code('="001283"') == "001283"
    assert normalize_stock_code("1283") == "001283"
    assert parse_ipc_subclasses("H01L21/3065; H01L21/3065; G06F1/00") == ["G06F", "H01L"]


def test_core_deduplicates_application_and_retains_firm_edges():
    records = [
        _record("CN1", "001283"),
        _record("CN1", "1283", relation="上市公司的子公司"),
        _record("CN1", "002443", relation="上市公司的联营企业"),
    ]

    core, edges, universe = canonicalize_records(records)

    assert len(core) == 1
    assert set(edges["stkcd"]) == {"001283", "002443"}
    assert core.loc[0, "source_record_count"] == 3
    assert universe.loc[0, "application_no"] == "CN1"
    assert universe.loc[0, "eligible_firm_count"] == 1


def test_same_firm_relation_union_keeps_any_baseline_eligible_evidence():
    records = [
        _record("CN1", "001283", relation="上市公司的联营企业"),
        _record("CN1", "001283", relation="上市公司本身"),
    ]

    core, edges, universe = canonicalize_records(records)

    assert len(core) == 1
    assert len(edges) == 1
    assert edges.loc[0, "baseline_eligible"] == 1
    assert "上市公司的联营企业" in edges.loc[0, "relation"]
    assert "上市公司本身" in edges.loc[0, "relation"]
    assert len(universe) == 1


def test_design_patent_is_retained_in_core_and_edges_but_excluded_from_baseline_universe():
    records = [
        _record("CN1", "001283", patent_type="发明申请"),
        _record("CN2", "001283", patent_type="外观设计", ipc=""),
    ]

    core, edges, universe = canonicalize_records(records)

    assert set(core["patent_type_group"]) == {"invention", "design"}
    assert set(edges["application_no"]) == {"CN1", "CN2"}
    assert set(universe["application_no"]) == {"CN1"}


def test_group_mapping_preserves_listed_parent_and_relation_type():
    records = [
        _record("CN1", "001283", relation="上市公司本身"),
        _record("CN1", "001283", relation="上市公司的子公司"),
    ]

    _, edges, universe = canonicalize_records(records)

    assert len(universe) == 1
    assert edges.loc[0, "listed_parent"] == "001283"
    assert "上市公司的子公司" in edges.loc[0, "relation_type"]
    assert edges.loc[0, "applicant"] == "applicant-001283"


def test_fractional_ipc_weights_sum_to_one_and_single_ipc_has_no_pair():
    records = [
        _record("CN1", "001283", ipc="A01B1/00; G06F1/00"),
        _record("CN2", "001283", application_date="1985-01-02", ipc="A01B1/00"),
    ]
    core, _, universe = canonicalize_records(records)
    pairs, ipc_edges = build_pair_events(universe, core)

    weights = ipc_edges.groupby("application_no")["fractional_weight"].sum()
    assert weights.to_dict() == {"CN1": pytest.approx(1.0), "CN2": pytest.approx(1.0)}
    assert set(pairs["application_no"]) == {"CN1"}
    assert pairs.loc[:, ["ipc_a", "ipc_b"]].iloc[0].tolist() == ["A01B", "G06F"]


def test_pair_history_is_same_day_order_invariant_and_strictly_prior():
    records = [
        _record("CN1", "001283", application_date="1985-01-01", ipc="A01B1/00; G06F1/00"),
        _record("CN2", "001283", application_date="1985-01-01", ipc="A01B1/00; G06F1/00"),
        _record("CN3", "001283", application_date="1985-01-02", ipc="A01B1/00; G06F1/00"),
    ]
    core, _, universe = canonicalize_records(records)
    pairs, _ = build_pair_events(universe, core)

    first = pairs.sort_values("application_no").reset_index(drop=True)
    reversed_universe = universe.iloc[::-1].reset_index(drop=True)
    reversed_pairs, _ = build_pair_events(reversed_universe, core)
    second = reversed_pairs.sort_values("application_no").reset_index(drop=True)

    assert first["historical_first"].tolist() == [1, 1, 0]
    assert first["historical_pair_count"].tolist() == [0, 0, 2]
    pd.testing.assert_frame_equal(
        first[["application_no", "historical_first", "historical_pair_count"]],
        second[["application_no", "historical_first", "historical_pair_count"]],
    )


def test_pair_rarity_is_finite_at_zero_history_and_falls_with_frequency():
    records = [
        _record("CN1", "001283", application_date="1985-01-01", ipc="A01B1/00; G06F1/00"),
        _record("CN2", "001283", application_date="1985-01-02", ipc="A01B1/00; G06F1/00"),
        _record("CN3", "001283", application_date="1985-01-03", ipc="A01B1/00; G06F1/00"),
    ]
    core, _, universe = canonicalize_records(records)
    pairs, _ = build_pair_events(universe, core)

    assert math.isfinite(pairs["pair_rarity"].iloc[0])
    assert pairs["pair_rarity"].iloc[0] > pairs["pair_rarity"].iloc[1]
    assert pairs["pair_rarity"].iloc[1] > pairs["pair_rarity"].iloc[2]


def test_quarterly_metrics_use_fractional_mass_and_pooled_four_quarter_inputs():
    records = [
        _record("CN1", "001283", application_date="1994-01-01", ipc="A01B1/00; G06F1/00"),
        _record("CN2", "001283", application_date="1994-04-01", ipc="A01B1/00"),
    ]
    core, _, universe = canonicalize_records(records)
    pairs, ipc_edges = build_pair_events(universe, core)
    metrics = aggregate_quarterly_metrics(universe, ipc_edges, pairs)

    q1 = metrics.loc[metrics["quarter"] == "1994Q1"].iloc[0]
    assert q1["patent_count"] == 1
    assert q1["invention_patent_count"] == 1
    assert q1["ipc_coded_patent_count"] == 1
    assert q1["multi_ipc_patent_count"] == 1
    assert q1["all_ipc_pair_count"] == 1
    assert q1["novel_ipc_pair_count"] == 1
    assert q1["ipc_hhi"] == pytest.approx(0.5)
    assert q1["ipc_entropy"] == pytest.approx(math.log(2))


def test_quarterly_metrics_include_scope_variety_distance_and_rao_stirling():
    records = [
        _record("CN1", "001283", application_date="1985-01-01", ipc="A01B1/00; G06F1/00"),
        _record("CN2", "001283", application_date="1994-01-01", ipc="A01B1/00; G06F1/00"),
    ]
    core, _, universe = canonicalize_records(records)
    pairs, ipc_edges = build_pair_events(universe, core)
    metrics = aggregate_quarterly_metrics(universe, ipc_edges, pairs)

    q1994 = metrics.loc[metrics["quarter"] == "1994Q1"].iloc[0]
    assert q1994["average_patent_scope"] == pytest.approx(2.0)
    assert q1994["median_patent_scope"] == pytest.approx(2.0)
    assert q1994["related_variety"] == pytest.approx(0.0)
    assert q1994["unrelated_variety"] == pytest.approx(math.log(2))
    assert q1994["average_technological_distance"] == pytest.approx(0.0)
    assert q1994["median_technological_distance"] == pytest.approx(0.0)
    assert q1994["rao_stirling_diversity"] == pytest.approx(0.0)


def _write_raw_csv(path, records):
    header = [f"column_{index}" for index in range(31)]
    lines = [",".join(header)]
    for values in records:
        stream = io.StringIO(newline="")
        csv.writer(stream).writerow(values)
        lines.append(stream.getvalue().rstrip("\r\n"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def _raw_values(application_no, application_date, ipc):
    values = [""] * 31
    values[0] = '="001283"'
    values[2] = "company"
    values[4] = "上市公司本身"
    values[9] = "发明申请"
    values[10] = "applicant"
    values[11] = "企业"
    values[16] = application_no
    values[17] = application_date
    values[25] = ipc
    values[26] = ipc.split(";")[0]
    values[28] = "multi-line\nabstract"
    return values


def test_small_end_to_end_run_writes_checkpoints_and_resumes(tmp_path):
    raw = tmp_path / "patents.csv"
    _write_raw_csv(
        raw,
        [
            _raw_values("CN1", "1985-01-01", "A01B1/00; G06F1/00"),
            _raw_values("CN1", "1985-01-01", "A01B1/00; G06F1/00"),
            _raw_values("CN2", "1994-01-01", "A01B1/00"),
        ],
    )
    dictionary = tmp_path / "dictionary.parquet"
    pd.DataFrame(
        [
            {"ipc_code": "A01B", "ipc_level": "subclass", "is_valid": True, "scheme_version": "2026.01"},
            {"ipc_code": "G06F", "ipc_level": "subclass", "is_valid": True, "scheme_version": "2026.01"},
        ]
    ).to_parquet(dictionary, index=False)
    output = tmp_path / "output"

    first = run_pipeline(raw, dictionary, output, shard_count=2, skip_plots=True)
    metrics = pd.read_parquet(first["metrics"])
    summary = json.loads((output / "parse_summary.json").read_text(encoding="utf-8"))
    assert summary["logical_records_seen"] == 3
    assert summary["successful_records"] == 3
    assert json.loads((output / "02_core_complete.json").read_text(encoding="utf-8"))["unique_application_count"] == 2
    assert (output / "01_parse_complete.json").exists()
    assert (output / "06_quarterly_metrics_complete.json").exists()
    assert metrics.loc[metrics["quarter"] == "1994Q1", "patent_count"].iloc[0] == 1

    second = run_pipeline(raw, dictionary, output, shard_count=2, skip_plots=True)
    pd.testing.assert_frame_equal(
        pd.read_parquet(first["metrics"]), pd.read_parquet(second["metrics"])
    )
