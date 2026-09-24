from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from preprocessing.patent_detail.canonical import (
    canonicalize_records,
    ipc_main_group,
    normalize_applicant_identity,
)
from subprojects.text_kpst.src.metrics import add_field_adjustment, compute_kpst_metrics
from subprojects.text_kpst.src.preprocess import PatentTextPreprocessor, normalize_patent_text
from subprojects.text_kpst.src.scalable import (
    _applicant_domain_keys,
    tokenize_and_build_vocabulary,
)
from subprojects.text_kpst.src.tfbidf import bidf_weight, build_pit_tfbidf


def test_applicant_identity_uses_name_and_address_and_strips_postcode():
    a = normalize_applicant_identity(
        "深圳市豪鹏科技有限公司",
        "518111 广东省深圳市龙岗区平湖镇山厦罗山工业区A2栋",
    )
    b = normalize_applicant_identity(
        "深圳市豪鹏科技有限公司",
        "广东省深圳市龙岗区平湖镇山厦罗山工业区A2栋",
    )
    assert a == b
    assert "518111" not in a


def test_ipc_main_group_keeps_hierarchy():
    assert ipc_main_group("H01M4/02") == "H01M4"
    assert ipc_main_group(" B23K 11/36 ") == "B23K11"


def test_preprocessor_preserves_technical_digits():
    processor = PatentTextPreprocessor(tokenizer=lambda text: text.split())
    assert normalize_patent_text("  ５G\n电池  ") == "5G 电池"
    assert processor.tokenize("5G 锂电池") == ["5G", "锂电池"]


def test_bidf_matches_appendix_equation_3():
    assert bidf_weight(100, 9) == pytest.approx(np.log(100 / 10))


def test_tfbidf_same_date_freeze_and_strict_prior():
    docs = [["电池"], ["芯片"], ["新词"]]
    dates = ["2000-01-01", "2000-01-01", "2000-01-02"]
    matrix, vocab, prior = build_pit_tfbidf(docs, dates)
    assert prior.tolist() == [0, 0, 2]
    assert matrix[0].nnz == 0
    assert matrix[1].nnz == 0
    assert matrix[2, vocab["新词"]] != 0


def test_bs_fs_denominator_uses_total_patent_count_not_surviving_candidates():
    matrix = sparse.csr_matrix(np.array([[1.0], [1.0], [1.0]], dtype=float))
    out = compute_kpst_metrics(
        matrix,
        application_years=[2000, 2001, 2002],
        applicant_keys=["A", "A", "B"],
        ipc_domain_keys=["X", "X", "X"],
        window_years=1,
        denominator_counts_by_year={2000: 1, 2001: 1, 2002: 1},
    ).set_index("patent_index")
    assert out.loc[1, "backward_similarity_sum"] == pytest.approx(0.0)
    assert out.loc[1, "backward_denominator_patents"] == 1
    assert out.loc[1, "average_backward_similarity"] == pytest.approx(0.0)


def test_method6_backward_same_ipc_forward_all_ipc():
    matrix = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
    )
    out = compute_kpst_metrics(
        matrix,
        application_years=[2000, 2001, 2002],
        applicant_keys=["A", "B", "C"],
        ipc_domain_keys=["X", "X", "Y"],
        window_years=1,
    ).set_index("patent_index")
    assert out.loc[1, "average_backward_similarity"] == pytest.approx(1.0)
    assert out.loc[1, "forward_denominator_patents"] == 1
    assert out.loc[1, "average_forward_similarity"] == pytest.approx(0.0)


def test_field_adjustment_is_field_mean_bs_over_overall_mean_bs():
    frame = pd.DataFrame(
        {
            "application_year": [2000, 2000, 2000, 2000],
            "ipc_domain_key": ["A", "A", "B", "B"],
            "average_backward_similarity": [1.0, 1.0, 2.0, 2.0],
            "kpst_ratio_unadjusted": [1.0, 1.0, 1.0, 1.0],
        }
    )
    out = add_field_adjustment(frame)
    assert out.loc[0, "field_similarity_adjustment"] == pytest.approx(1 / 1.5)
    assert out.loc[2, "field_similarity_adjustment"] == pytest.approx(2 / 1.5)


def test_canonicalization_keeps_longest_text_and_baseline_relation():
    rows = [
        {
            "application_no": "CN1",
            "application_date": "2011-07-18",
            "publication_no": "P1",
            "publication_date": "2012-01-01",
            "grant_no": "G1",
            "grant_date": "2013-01-01",
            "patent_type": "发明申请",
            "patent_type_group": "invention",
            "patent_title": "电池",
            "abstract": "短摘要",
            "main_claim": "权利要求",
            "applicant": "申请人",
            "applicant_type": "企业",
            "applicant_address": "100000 北京市海淀区某路1号",
            "applicant_region": "中国",
            "applicant_city": "北京市",
            "applicant_district": "海淀区",
            "applicant_identity_key": "申请人||北京市海淀区某路1号",
            "inventors": "甲",
            "ipc_all_raw": "H01M4/02",
            "ipc_main_raw": "H01M4/02",
            "ipc_codes": ["H01M4/02"],
            "ipc_main_code": "H01M4/02",
            "ipc_subclass": "H01M",
            "ipc_domain_key": "H01M4",
            "stkcd": "001283",
            "company_name": "公司",
            "relation": "上市公司本身",
            "baseline_firm_eligible": 1,
            "is_granted_invention": 1,
        },
        {
            "application_no": "CN1",
            "application_date": "2011-07-18",
            "publication_no": "P1",
            "publication_date": "2012-01-01",
            "grant_no": "G1",
            "grant_date": "2013-01-01",
            "patent_type": "发明申请",
            "patent_type_group": "invention",
            "patent_title": "电池",
            "abstract": "这是更长的摘要",
            "main_claim": "权利要求",
            "applicant": "申请人",
            "applicant_type": "企业",
            "applicant_address": "100000 北京市海淀区某路1号",
            "applicant_region": "中国",
            "applicant_city": "北京市",
            "applicant_district": "海淀区",
            "applicant_identity_key": "申请人||北京市海淀区某路1号",
            "inventors": "甲",
            "ipc_all_raw": "H01M4/02",
            "ipc_main_raw": "H01M4/02",
            "ipc_codes": ["H01M4/02"],
            "ipc_main_code": "H01M4/02",
            "ipc_subclass": "H01M",
            "ipc_domain_key": "H01M4",
            "stkcd": "001283",
            "company_name": "公司",
            "relation": "上市公司的子公司",
            "baseline_firm_eligible": 1,
            "is_granted_invention": 1,
        },
    ]
    canonical, edges = canonicalize_records(rows)
    assert len(canonical) == 1
    assert canonical.loc[0, "abstract"] == "这是更长的摘要"
    assert canonical.loc[0, "is_granted_invention"] == 1
    assert len(edges) == 1



def test_missing_applicant_identity_never_forms_backward_composite_key():
    keys = _applicant_domain_keys(["", "A"], ["H01M4", "H01M4"])
    assert keys.tolist() == ["", "A||DOMAIN||H01M4"]


def test_vector_stage_cleans_stale_files_and_audits_text_coverage(tmp_path):
    canonical_root = tmp_path / "canonical"
    year_dir = canonical_root / "application_year=2000"
    year_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "application_no": ["CN1", "CN2"],
            "application_date": ["2000-01-01", "2000-02-01"],
            "application_year": [2000, 2000],
            "is_granted_invention": [1, 1],
            "patent_title": ["alpha beta", ""],
            "abstract": ["", ""],
            "applicant_identity_key": ["A||ADDR", ""],
            "applicant_address": ["ADDR", ""],
            "ipc_domain_key": ["H01M4", "H01M4"],
            "baseline_firm_eligible": [1, 1],
        }
    ).to_parquet(year_dir / "part.parquet", index=False)

    vector_root = tmp_path / "vectors"
    vector_root.mkdir()
    stale = vector_root / "stale.parquet"
    stale.write_text("old", encoding="utf-8")

    _, counts = tokenize_and_build_vocabulary(
        canonical_root,
        vector_root,
        text_fields=["patent_title", "abstract"],
        min_corpus_frequency=1,
    )

    assert not stale.exists()
    assert counts == {2000: 1}
    audit = pd.read_csv(vector_root / "text_coverage_by_year.csv")
    assert audit.loc[0, "granted_invention_count"] == 2
    assert audit.loc[0, "vectorizable_patent_count"] == 1
    assert audit.loc[0, "text_coverage_rate"] == pytest.approx(0.5)
