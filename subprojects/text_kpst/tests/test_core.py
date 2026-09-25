from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing.patent_detail.canonical import (
    applicant_key,
    application_quarter,
    ipc_main_group,
)
from subprojects.text_kpst.src.metrics import add_field_adjustment
from subprojects.text_kpst.src.scalable import _window_quarters
from subprojects.text_kpst.src.tfbidf import bidf


def test_bidf_formula():
    assert bidf(100, 9) == pytest.approx(np.log(10))


def test_keys_and_quarter():
    assert applicant_key("深圳市豪鹏科技有限公司", "518111 广东省深圳市龙岗区A2栋") == applicant_key(
        "深圳市豪鹏科技有限公司", "广东省深圳市龙岗区A2栋"
    )
    assert ipc_main_group("H01M4/02") == "H01M4"
    assert application_quarter("2011-07-18") == "2011Q3"


def test_eight_quarter_window():
    backward, forward = _window_quarters("2024Q1", 8)
    assert backward == [
        "2022Q1", "2022Q2", "2022Q3", "2022Q4",
        "2023Q1", "2023Q2", "2023Q3", "2023Q4",
    ]
    assert forward == [
        "2024Q2", "2024Q3", "2024Q4", "2025Q1",
        "2025Q2", "2025Q3", "2025Q4", "2026Q1",
    ]


def test_quarterly_field_adjustment():
    frame = pd.DataFrame({
        "application_quarter": ["2000Q1"] * 4,
        "ipc_main_group": ["A", "A", "B", "B"],
        "bs": [1.0, 1.0, 2.0, 2.0],
        "fs": [1.0, 1.0, 2.0, 2.0],
    })
    out = add_field_adjustment(frame)
    assert out.loc[0, "field_adjustment"] == pytest.approx(1 / 1.5)
    assert out.loc[2, "field_adjustment"] == pytest.approx(2 / 1.5)
