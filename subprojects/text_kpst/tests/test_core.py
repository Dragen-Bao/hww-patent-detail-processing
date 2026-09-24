from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from preprocessing.patent_detail.canonical import applicant_key, ipc_main_group
from subprojects.text_kpst.src.metrics import add_field_adjustment
from subprojects.text_kpst.src.tfbidf import bidf


def test_bidf_formula():
    assert bidf(100, 9) == pytest.approx(np.log(10))


def test_applicant_key_and_ipc_group():
    assert applicant_key("深圳市豪鹏科技有限公司", "518111 广东省深圳市龙岗区A2栋") == applicant_key(
        "深圳市豪鹏科技有限公司", "广东省深圳市龙岗区A2栋"
    )
    assert ipc_main_group("H01M4/02") == "H01M4"


def test_method6_similarity_logic():
    target = sparse.csr_matrix([[1.0, 0.0]])
    same_field = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    all_forward = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    bs = float((target @ same_field.T).sum()) / 2
    fs = float((target @ all_forward.T).sum()) / 2
    assert bs == pytest.approx(0.5)
    assert fs == pytest.approx(0.5)


def test_field_adjustment():
    frame = pd.DataFrame({
        "application_year": [2000, 2000, 2000, 2000],
        "ipc_main_group": ["A", "A", "B", "B"],
        "bs": [1.0, 1.0, 2.0, 2.0],
        "fs": [1.0, 1.0, 2.0, 2.0],
    })
    out = add_field_adjustment(frame)
    assert out.loc[0, "field_adjustment"] == pytest.approx(1 / 1.5)
    assert out.loc[2, "field_adjustment"] == pytest.approx(2 / 1.5)
