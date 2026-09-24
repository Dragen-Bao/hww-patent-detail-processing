from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from preprocessing.patent_detail.text_pipeline import (
    aggregate_group_quarter,
    aggregate_market_quarter,
    build_pit_text_metrics,
    canonicalize_text_records,
    char_ngram_counts,
    clean_abstract,
    iter_full_logical_records,
    run_text_extraction,
    run_text_metrics,
    score_text_batch,
)


def _text_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_clean_abstract_normalizes_spacing_and_unicode():
    assert clean_abstract("  电池\n\t系统，  高效  ") == "电池 系统, 高效"


def test_char_ngrams_are_deterministic_and_count_overlapping_windows():
    counts = char_ngram_counts("电池电池", min_n=2, max_n=2)
    assert counts == Counter({"电池": 2, "池电": 1})


def test_full_logical_parser_preserves_quoted_multiline_abstract(tmp_path):
    fields = [""] * 31
    fields[0] = '="001283"'
    fields[2] = "公司"
    fields[4] = "上市公司本身"
    fields[10] = "申请人"
    fields[11] = "企业"
    fields[9] = "发明申请"
    fields[16] = "CN1"
    fields[17] = "2020-01-01"
    fields[25] = "A01B 1/02"
    fields[26] = "A01B 1/02"
    fields[28] = "第一行摘要\n第二行摘要"
    path = tmp_path / "raw.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write(",".join(fields[:28]) + ',"第一行摘要\n第二行摘要",,\n')
    rows = list(iter_full_logical_records(path))
    assert len(rows) == 1
    assert rows[0][16] == "CN1"
    assert "第二行摘要" in rows[0][28]


def test_full_logical_parser_preserves_unquoted_continuation_after_complete_first_line(tmp_path):
    fields = [""] * 31
    fields[0] = '="001283"'
    fields[2] = "公司"
    fields[4] = "上市公司本身"
    fields[9] = "发明申请"
    fields[16] = "CN1"
    fields[17] = "2020-01-01"
    fields[28] = "第一行摘要"
    path = tmp_path / "raw_unquoted.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write(",".join(fields[:28]) + ",第一行摘要,,\n第二行摘要\n")
    rows = list(iter_full_logical_records(path))
    assert len(rows) == 1
    assert "第一行摘要" in rows[0][28]
    assert "第二行摘要" in rows[0][28]


def test_build_pit_text_metrics_excludes_same_day_and_self_from_history():
    patents = _text_frame(
        [
            {"application_no": "A", "application_date": "2020-01-01", "abstract_clean": "电池系统"},
            {"application_no": "B", "application_date": "2020-01-01", "abstract_clean": "电池系统"},
            {"application_no": "C", "application_date": "2020-01-02", "abstract_clean": "电池系统"},
        ]
    )
    out = build_pit_text_metrics(patents)
    assert out.set_index("application_no").loc["A", "abstract_history_document_count"] == 0
    assert out.set_index("application_no").loc["B", "abstract_history_document_count"] == 0
    assert out.set_index("application_no").loc["C", "abstract_history_document_count"] == 2
    assert pd.isna(out.set_index("application_no").loc["A", "abstract_prior_centroid_novelty"])
    assert out.set_index("application_no").loc["C", "abstract_unseen_ngram_share"] == pytest.approx(0.0)


def test_canonical_core_uses_baseline_metadata_when_nonbaseline_row_comes_first():
    def record(relation, eligible):
        return {
            "stkcd_raw": '="001283"',
            "stkcd": "001283",
            "company_name": "公司",
            "relation": relation,
            "relation_type": relation,
            "listed_parent": "001283",
            "applicant": relation,
            "applicant_type": "企业",
            "application_no": "CN1",
            "application_date": "2020-01-01",
            "quarter": "2020Q1",
            "patent_type_group": "invention",
            "baseline_eligible": eligible,
            "abstract_clean": "电池系统",
            "claim_present_flag": 0,
            "claim_length_chars": 0,
        }

    core, _ = canonicalize_text_records([record("上市公司的联营企业", 0), record("上市公司本身", 1)])
    assert core.loc[0, "relation"] == "上市公司本身"
    assert "上市公司的联营企业" in core.loc[0, "relation_type_values"]


def test_text_extraction_writes_application_and_group_partitions(tmp_path):
    def row(application_no, abstract, relation="上市公司本身"):
        values = [""] * 31
        values[0] = '="001283"'
        values[2] = "公司"
        values[4] = relation
        values[9] = "发明申请"
        values[10] = "申请人"
        values[11] = "企业"
        values[16] = application_no
        values[17] = "2020-01-01"
        values[25] = "A01B 1/02"
        values[26] = "A01B 1/02"
        values[28] = abstract
        return values

    raw = tmp_path / "raw.csv"
    with raw.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = __import__("csv").writer(handle, lineterminator="\n")
        writer.writerow([f"c{i}" for i in range(31)])
        writer.writerow(row("CN1", "第一项摘要"))
        writer.writerow(row("CN1", "第一项摘要的更长版本"))
        writer.writerow(row("CN1", "第一项摘要的更长版本", relation="上市公司的子公司"))
        writer.writerow(row("CN2", "第二项摘要", relation="联营企业"))
    out = tmp_path / "out"
    summary = run_text_extraction(raw, out, shard_count=2, buffer_records=1)
    assert summary["application_count"] == 2
    core = pd.concat([pd.read_parquet(path) for path in (out / "text_core").rglob("*.parquet")])
    assert set(core["application_no"]) == {"CN1", "CN2"}
    assert core.loc[core["application_no"].eq("CN1"), "abstract_length_chars"].iloc[0] == len("第一项摘要的更长版本")
    cn1 = core.loc[core["application_no"].eq("CN1")].iloc[0]
    assert "上市公司本身" in cn1["relation_type_values"]
    assert "上市公司的子公司" in cn1["relation_type_values"]
    metrics_summary = run_text_metrics(out, out / "metrics", minimum_group_patents=2)
    assert metrics_summary["patent_count"] == 1
    market = pd.read_parquet(out / "metrics" / "quarterly_text_innovation_metrics.parquet")
    assert market.loc[0, "patent_count"] == 1


def test_pit_text_batch_scores_before_updating_history():
    prior_counts = char_ngram_counts("电池系统")
    history = {
        "document_count": 1,
        "document_frequency": Counter({token: 1 for token in prior_counts}),
        "term_sum": Counter({token: 1.0 for token in prior_counts}),
    }
    batch = _text_frame(
        [
            {"application_no": "A", "application_date": "2020-01-01", "abstract_clean": "电池系统"},
            {"application_no": "B", "application_date": "2020-01-01", "abstract_clean": "电池系统"},
        ]
    )
    scored, updated = score_text_batch(batch, history)
    assert scored.loc[0, "abstract_history_document_count"] == 1
    assert scored.loc[1, "abstract_history_document_count"] == 1
    assert scored.loc[0, "abstract_unseen_ngram_share"] == pytest.approx(0.0)
    assert scored.loc[1, "abstract_unseen_ngram_share"] == pytest.approx(0.0)
    assert updated["document_count"] == 3


def test_market_aggregation_keeps_missing_text_out_of_text_denominator():
    patents = _text_frame(
        [
            {"quarter": "2020Q1", "application_no": "A", "abstract_clean": "x", "abstract_prior_centroid_novelty": 0.2},
            {"quarter": "2020Q1", "application_no": "B", "abstract_clean": "", "abstract_prior_centroid_novelty": float("nan")},
        ]
    )
    out = aggregate_market_quarter(patents)
    assert out.loc[0, "patent_count"] == 2
    assert out.loc[0, "abstract_valid_count"] == 1
    assert out.loc[0, "abstract_coverage_rate"] == pytest.approx(0.5)
    assert out.loc[0, "abstract_prior_centroid_novelty_mean"] == pytest.approx(0.2)


def test_group_aggregation_has_low_sample_flag_and_no_duplicate_application_counts():
    patents = _text_frame(
        [
            {"quarter": "2020Q1", "application_no": "A", "listed_parent": "001", "abstract_clean": "x", "abstract_prior_centroid_novelty": 0.2},
            {"quarter": "2020Q1", "application_no": "A", "listed_parent": "001", "abstract_clean": "x", "abstract_prior_centroid_novelty": 0.2},
        ]
    )
    out = aggregate_group_quarter(patents, minimum_patents=2)
    assert out.loc[0, "patent_count"] == 1
    assert out.loc[0, "low_sample_flag"] == 1
