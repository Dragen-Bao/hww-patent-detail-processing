from pathlib import Path

import pandas as pd


def test_logical_record_scanner_keeps_multiline_text_in_one_record(tmp_path: Path):
    from src.audit_patent_ipc import iter_logical_records

    path = tmp_path / "patents.csv"
    path.write_text(
        "关联股票代码,code,IPC分类号,IPC主分类号,摘要文本\n"
        "=\"\"001283\"\",1283,H01M4/02; H01M4/04,H01M4/02,第一行\n"
        "摘要续行\n"
        "=\"\"002443\"\",2443,H01L21/3065,H01L21/3065,第二条\n",
        encoding="utf-8",
    )
    rows = list(iter_logical_records(path))
    assert len(rows) == 2
    assert rows[0][4].endswith("摘要续行")


def test_iter_observed_ipc_splits_all_and_main_fields(tmp_path: Path):
    from src.audit_patent_ipc import iter_observed_ipc

    path = tmp_path / "patents.csv"
    path.write_text(
        "关联股票代码,code,IPC分类号,IPC主分类号,摘要文本\n"
        "=\"\"002443\"\",2443,H01L21/3065; H01M4/02,H01L21/3065,第二条\n",
        encoding="utf-8",
    )
    rows = list(iter_observed_ipc(path))
    assert {row["ipc_code"] for row in rows} == {"H01L21/3065", "H01M4/02"}
    assert sum(row["is_main"] for row in rows) == 1


def test_audit_marks_code_found_only_in_historical_dictionary(tmp_path: Path):
    from src.audit_patent_ipc import audit_observed_ipc

    patent_path = tmp_path / "patents.csv"
    patent_path.write_text(
        "关联股票代码,code,IPC分类号,IPC主分类号,摘要文本\n"
        "=\"\"002443\"\",2443,H01L21/3065,H01L21/3065,第二条\n",
        encoding="utf-8",
    )
    dictionary_path = tmp_path / "dictionary.parquet"
    pd.DataFrame(
        [
            {
                "ipc_code": "H01L21/3065",
                "scheme_version": "2025.01",
                "is_valid": True,
                "ipc_level": "subgroup",
            }
        ]
    ).to_parquet(dictionary_path, index=False)
    output_dir = tmp_path / "audit"
    audit_observed_ipc(patent_path, dictionary_path, output_dir)
    coverage = pd.read_csv(output_dir / "observed_ipc_coverage.csv", encoding="utf-8-sig")
    assert set(coverage["status"]) == {"valid_in_2025_only"}
