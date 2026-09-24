from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from subprojects.text_kpst.src.metrics import compute_kpst_metrics
from subprojects.text_kpst.src.preprocess import PatentTextPreprocessor, normalize_patent_text
from subprojects.text_kpst.src.tfbidf import build_pit_tfbidf


def test_normalize_preserves_technical_digits_and_normalizes_width():
    assert normalize_patent_text("  ５G\n电池  ") == "5G 电池"


def test_preprocessor_accepts_explicit_tokenizer_without_jieba_dependency():
    processor = PatentTextPreprocessor(
        stopwords={"一种"},
        tokenizer=lambda text: text.split(),
    )
    assert processor.tokenize("一种 锂电池 隔膜") == ["锂电池", "隔膜"]


def test_tfbidf_uses_only_strictly_prior_years():
    documents = [
        ["电池", "隔膜"],
        ["电池", "芯片"],
        ["电池", "芯片"],
    ]
    years = [2000, 2001, 2001]
    matrix, vocabulary, prior_counts = build_pit_tfbidf(documents, years)
    assert prior_counts.tolist() == [0, 1, 1]
    assert matrix[0].nnz == 0
    assert matrix[1, vocabulary["芯片"]] > 0
    assert matrix[2, vocabulary["芯片"]] > 0


def test_kpst_backward_same_ipc_forward_all_ipc():
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
    )
    out = compute_kpst_metrics(
        matrix,
        application_years=[2000, 2001, 2002, 2003],
        applicant_keys=["A", "B", "C", "D"],
        ipc_domain_keys=["X", "X", "X", "Y"],
        window_years=2,
    ).set_index("patent_index")

    assert out.loc[1, "average_backward_similarity"] == pytest.approx(1.0)
    assert out.loc[1, "forward_comparison_count"] == 2
    assert out.loc[1, "average_forward_similarity"] == pytest.approx(0.5)


def test_same_applicant_is_removed_from_comparison_pool():
    matrix = sparse.csr_matrix(np.asarray([[1.0], [1.0]]))
    out = compute_kpst_metrics(
        matrix,
        application_years=[2000, 2001],
        applicant_keys=["A", "A"],
        ipc_domain_keys=["X", "X"],
        window_years=5,
    ).set_index("patent_index")
    assert out.loc[1, "backward_comparison_count"] == 0
    assert np.isnan(out.loc[1, "average_backward_similarity"])
