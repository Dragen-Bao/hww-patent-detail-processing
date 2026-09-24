from pathlib import Path
import zipfile


def write_fixture_zip(path: Path, member_name: str = "fixture.xml") -> None:
    fixture = Path(__file__).parent / "fixtures" / "wipo_scheme_fixture.xml"
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(fixture, member_name)


def test_normalize_ipc_code_removes_formatting_only():
    from src.ipc_dictionary import normalize_ipc_code

    assert normalize_ipc_code(" H01L21/3065 [2026.01] ") == "H01L21/3065"
    assert normalize_ipc_code("H01L 21/00") == "H01L21/00"
    assert normalize_ipc_code("H01L0021306500") == "H01L21/3065"


def test_parent_chain_for_subgroup():
    from src.ipc_dictionary import ipc_level, parent_ipc_code

    code = "H01L21/3065"
    assert ipc_level(code) == "subgroup"
    assert parent_ipc_code(code) == "H01L21/00"
    assert parent_ipc_code("H01L21/00") == "H01L"
    assert parent_ipc_code("H01L") == "H01"
    assert parent_ipc_code("H01") == "H"
    assert parent_ipc_code("H") is None


def test_parse_wipo_fixture_reads_titles_and_levels(tmp_path: Path):
    from src.ipc_dictionary import parse_wipo_scheme

    archive = tmp_path / "scheme.zip"
    write_fixture_zip(archive)
    rows = parse_wipo_scheme(archive)
    by_code = {row["ipc_code"]: row for row in rows}
    assert by_code["H"]["title_en"] == "ELECTRICITY"
    assert by_code["H01L21/3065"]["ipc_level"] == "subgroup"


def test_parse_cnipa_titles_handles_wrapped_title():
    from src.ipc_dictionary import parse_cnipa_text

    text = (Path(__file__).parent / "fixtures" / "cnipa_text_fixture.txt").read_text(encoding="utf-8")
    rows = parse_cnipa_text(text)
    by_code = {row["ipc_code"]: row for row in rows}
    assert by_code["H"]["title_zh"] == "电学"
    assert by_code["H01L21/3065"]["title_zh"] == "更具体技术"


def test_parse_wipo_title_list_normalizes_padded_symbols(tmp_path: Path):
    from src.ipc_dictionary import parse_wipo_title_list

    archive = tmp_path / "title_list.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "EN_ipc_section_H_title_list_20260101.txt",
            "H\tELECTRICITY\nH01L0021306500\tPLASMA ETCHING\n",
        )
    rows = parse_wipo_title_list(archive)
    assert {row["ipc_code"]: row["title_en"] for row in rows}["H01L21/3065"] == "PLASMA ETCHING"
