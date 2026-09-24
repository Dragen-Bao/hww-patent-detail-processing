from pathlib import Path


def test_source_specs_include_structured_and_chinese_sources():
    from src.ipc_sources import SOURCE_SPECS

    names = {item["name"] for item in SOURCE_SPECS}
    assert {
        "wipo_scheme",
        "wipo_valid_symbols",
        "wipo_concordance",
        "wipo_title_list",
        "wipo_2025_scheme",
        "wipo_2025_valid_symbols",
    } <= names
    assert {"cnipa_A", "cnipa_H", "cnipa_2025_A", "cnipa_2025_H"} <= names


def test_sha256_file_returns_digest(tmp_path: Path):
    from src.ipc_sources import sha256_file

    path = tmp_path / "source.bin"
    path.write_bytes(b"ipc")
    assert sha256_file(path) == "18640e38f277a2b6ecbd7db448ed0147f162a291f681b473a627f269fd979811"
