"""Parse official IPC master files and Chinese classification text."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET


WIPO_NAMESPACE = "http://www.wipo.int/classifications/ipc/masterfiles"
_WIPO_TAG = "{" + WIPO_NAMESPACE + "}"
_PADDED_SYMBOL_RE = re.compile(r"^([A-H])(\d{2})([A-Z])(\d{4})(\d{6})$")
_DISPLAY_SYMBOL_RE = re.compile(
    r"^([A-H])(\d{2})([A-Z])(?:(\d{1,4})/(\d{1,6}))?$"
)
_CNIPA_SYMBOL_PATTERNS = (
    re.compile(r"^([A-H]\s*\d{2}\s*[A-Z]\s*\d{1,4}\s*/\s*\d{1,6})\s*(.*)$"),
    re.compile(r"^([A-H]\s*\d{2}\s*[A-Z])$"),
    re.compile(r"^([A-H]\s*\d{2})$"),
    re.compile(r"^([A-H]\s*\d{2}\s*[A-Z])\s+(.*)$"),
    re.compile(r"^([A-H]\s*\d{2})\s+(.*)$"),
    re.compile(r"^([A-H])$"),
    re.compile(r"^([A-H])(?:\s+)(.*)$"),
)
_VERSION_RE = re.compile(r"\[\s*\d{4}\.\d{2}\s*\]")


def normalize_ipc_code(value: str | None) -> str | None:
    """Normalize display or WIPO's zero-padded IPC symbol to one display form."""

    if value is None:
        return None
    text = _VERSION_RE.sub("", str(value)).strip().upper().replace("／", "/")
    text = re.sub(r"\s+", "", text)
    if not text or not re.fullmatch(r"[A-H].*", text):
        return None
    if len(text) == 1 and text in "ABCDEFGH":
        return text
    if re.fullmatch(r"[A-H]\d{2}", text):
        return text
    if re.fullmatch(r"[A-H]\d{2}[A-Z]", text):
        return text

    padded = _PADDED_SYMBOL_RE.fullmatch(text)
    if padded:
        section, class_number, subclass, main_raw, subgroup_raw = padded.groups()
        main = str(int(main_raw))
        subgroup = subgroup_raw.rstrip("0")
        if not subgroup:
            subgroup = "00"
        else:
            subgroup = str(int(subgroup)).zfill(2)
        return f"{section}{class_number}{subclass}{main}/{subgroup}"

    display = _DISPLAY_SYMBOL_RE.fullmatch(text)
    if not display:
        return None
    section, class_number, subclass, main_raw, subgroup_raw = display.groups()
    if main_raw is None:
        return f"{section}{class_number}{subclass}"
    subgroup = str(int(subgroup_raw)).zfill(2)
    return f"{section}{class_number}{subclass}{int(main_raw)}/{subgroup}"


def ipc_level(code: str) -> str:
    """Return the IPC hierarchy level for a normalized code."""

    normalized = normalize_ipc_code(code)
    if normalized is None:
        raise ValueError(f"Invalid IPC code: {code!r}")
    if len(normalized) == 1:
        return "section"
    if len(normalized) == 3:
        return "class"
    if len(normalized) == 4:
        return "subclass"
    if normalized.endswith("/00"):
        return "main_group"
    return "subgroup"


def parent_ipc_code(code: str) -> str | None:
    """Return the immediate parent in the IPC hierarchy."""

    normalized = normalize_ipc_code(code)
    if normalized is None:
        raise ValueError(f"Invalid IPC code: {code!r}")
    level = ipc_level(normalized)
    if level == "section":
        return None
    if level == "class":
        return normalized[0]
    if level == "subclass":
        return normalized[:3]
    if level == "main_group":
        return normalized[:4]
    return normalized.split("/", 1)[0] + "/00"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _zip_xml_bytes(zip_path: Path, name_hint: str) -> bytes:
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        preferred = [name for name in names if name_hint.lower() in name.lower()]
        if preferred:
            return archive.read(preferred[0])
        if len(names) == 1:
            return archive.read(names[0])
        raise FileNotFoundError(f"No XML member matching {name_hint!r} in {zip_path}")


def _direct_title(entry: ET.Element) -> str:
    title_parts: list[str] = []
    for title in entry.iter():
        if _local_name(title.tag) != "title":
            continue
        if title.text and title.attrib.get("lang", "en").lower() == "en":
            title_parts.append(title.text)
        for child in title:
            if _local_name(child.tag) != "titlePart":
                continue
            for text_node in child:
                if _local_name(text_node.tag) == "text" and text_node.text:
                    title_parts.append(text_node.text)
        break
    return re.sub(r"\s+", " ", " ".join(title_parts)).strip()


def parse_wipo_scheme(zip_path: Path) -> list[dict[str, str | None]]:
    """Parse English WIPO scheme entries into normalized hierarchy rows."""

    root = ET.fromstring(_zip_xml_bytes(zip_path, "EN_ipc_scheme_"))
    rows: list[dict[str, str | None]] = []
    included_kinds = {
        "s": "section",
        "c": "class",
        "u": "subclass",
        "m": "main_group",
        "g": "main_group",
        **{str(depth): "subgroup" for depth in range(1, 10)},
    }

    # Keep the parser useful for compact fixtures and equivalent XML exports
    # that represent entries as classification-item nodes.
    compact_items = [node for node in root.iter() if _local_name(node.tag) == "classification-item"]
    if compact_items:
        for node in compact_items:
            code = normalize_ipc_code(node.attrib.get("symbol"))
            if not code:
                continue
            level = node.attrib.get("level") or ipc_level(code)
            rows.append(
                {
                    "code": code,
                    "level": level,
                    "parent_code": normalize_ipc_code(node.attrib.get("parent")) or parent_ipc_code(code),
                    "title_en": _direct_title(node),
                    "source_symbol": node.attrib.get("symbol"),
                    "source_kind": level,
                }
            )
        return [
            _with_compatibility_keys(row)
            for row in sorted(rows, key=lambda row: str(row["code"]))
        ]

    def visit(node: ET.Element, inherited_parent: str | None) -> None:
        parent = inherited_parent
        if _local_name(node.tag) == "ipcEntry":
            kind = node.attrib.get("kind", "")
            code = normalize_ipc_code(node.attrib.get("symbol"))
            if kind in included_kinds and code:
                rows.append(
                    {
                        "code": code,
                        "level": included_kinds[kind],
                        "parent_code": parent or parent_ipc_code(code),
                        "title_en": _direct_title(node),
                        "source_symbol": node.attrib.get("symbol"),
                        "source_kind": kind,
                    }
                )
                parent = code
        for child in node:
            visit(child, parent)

    visit(root, None)

    # Core and advanced main-group entries can expose the same code. Retain
    # one row and prefer the entry with a useful/longer title.
    selected: dict[str, dict[str, str | None]] = {}
    for row in rows:
        code = str(row["code"])
        old = selected.get(code)
        if old is None or (not old["title_en"] and row["title_en"]) or len(str(row["title_en"])) > len(str(old["title_en"])):
            selected[code] = row
    return [
        _with_compatibility_keys(row)
        for row in sorted(selected.values(), key=lambda row: (str(row["code"]), str(row["source_kind"])))
    ]


def _with_compatibility_keys(row: dict[str, str | None]) -> dict[str, str | None]:
    """Expose concise internal names plus the names used by audit tables."""

    return {
        **row,
        "ipc_code": row["code"],
        "ipc_level": row["level"],
    }


def parse_wipo_valid_symbols(zip_path: Path) -> set[str]:
    """Read WIPO's official valid-symbol list and normalize its symbols."""

    root = ET.fromstring(_zip_xml_bytes(zip_path, "valid_symbols"))
    symbols: set[str] = set()
    for node in root.iter():
        if _local_name(node.tag).lower() == "ipcsymbol":
            code = normalize_ipc_code(node.attrib.get("symbol"))
            if code:
                symbols.add(code)
    return symbols


def parse_wipo_title_list(zip_path: Path) -> list[dict[str, str | None]]:
    """Read WIPO's tab-separated English title-list files."""

    rows: list[dict[str, str | None]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".txt"):
                continue
            for raw_line in archive.read(name).decode("utf-8-sig").splitlines():
                if "\t" not in raw_line:
                    continue
                raw_code, title = raw_line.split("\t", 1)
                code = normalize_ipc_code(raw_code)
                if code:
                    rows.append(
                        {
                            "ipc_code": code,
                            "title_en": re.sub(r"\s+", " ", title).strip(),
                        }
                    )
    return rows


def parse_wipo_concordance(zip_path: Path) -> list[dict[str, str | None]]:
    """Read version-to-version reclassification mappings from WIPO."""

    root = ET.fromstring(_zip_xml_bytes(zip_path, "concordancelist"))
    from_version = root.attrib.get("from-version")
    to_version = root.attrib.get("to-version")
    rows: list[dict[str, str | None]] = []
    for node in root.iter():
        if _local_name(node.tag) != "concordance":
            continue
        from_code = normalize_ipc_code(node.attrib.get("from-symbol"))
        if not from_code:
            continue
        to_nodes = [child for child in node if _local_name(child.tag) == "concordance-to"]
        for to_node in to_nodes:
            to_code = normalize_ipc_code(to_node.attrib.get("to-symbol"))
            if to_code:
                rows.append(
                    {
                        "from_version": from_version,
                        "to_version": to_version,
                        "from_code": from_code,
                        "to_code": to_code,
                        "modification": to_node.attrib.get("modification"),
                        "revision_project": to_node.attrib.get("revision-project"),
                    }
                )
    return rows


def _clean_cnipa_title(text: str) -> str:
    text = _VERSION_RE.sub("", text)
    text = re.sub(r"^[A-H]\s*(?:部|SECTION)\s*[-—–:]+\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*[.·。]+\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_cnipa_text(text: str) -> list[dict[str, str | None]]:
    """Parse the code/title lines from extracted CNIPA PDF text.

    CNIPA's PDF frequently wraps a title onto one or more following lines, so
    continuation text is attached to the most recent IPC code.
    """

    records: list[dict[str, str | None]] = []
    current: dict[str, str | None] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or re.fullmatch(r"\d+", line) or re.search(r"第\s*\d+\s*页", line):
            continue
        # Remove table-of-contents dot leaders while retaining the code/title
        # prefix on an actual section heading.
        if re.search(r"\.{4,}", line):
            line = re.sub(r"\.{4,}.*$", "", line).strip()
            if not line:
                continue
        match = next((pattern.match(line) for pattern in _CNIPA_SYMBOL_PATTERNS if pattern.match(line)), None)
        code = normalize_ipc_code(match.group(1)) if match else None
        candidate_title = match.group(2).strip() if match and match.lastindex and match.lastindex >= 2 else ""
        if code and len(code) == 1 and candidate_title and not re.match(
            r"^[A-H]\s*(?:部|SECTION)", candidate_title, flags=re.I
        ):
            code = None
        if code:
            if current is not None:
                current["title_zh"] = _clean_cnipa_title(str(current["title_zh"] or ""))
                records.append(current)
            title = candidate_title
            if ipc_level(code) == "section":
                title = re.sub(r"^[A-H]\s*(?:部|SECTION)\s*[-—–:]+\s*", "", title, flags=re.I)
            current = {
                "code": code,
                "level": ipc_level(code),
                "parent_code": parent_ipc_code(code),
                "title_zh": title,
            }
        elif current is not None and re.match(
            r"^(?:附注|注释|注|小类包括|本小类|本大类|本部|本分部|定义|NOTE|NOTES)",
            line,
            flags=re.I,
        ):
            current["title_zh"] = _clean_cnipa_title(str(current["title_zh"] or ""))
            records.append(current)
            current = None
        elif current is not None and not _VERSION_RE.fullmatch(line):
            current["title_zh"] = ((str(current["title_zh"] or "") + " " + line).strip())
    if current is not None:
        current["title_zh"] = _clean_cnipa_title(str(current["title_zh"] or ""))
        records.append(current)

    selected: dict[str, dict[str, str | None]] = {}
    for row in records:
        if not row["code"]:
            continue
        code = str(row["code"])
        old = selected.get(code)
        if old is None or (not old["title_zh"] and row["title_zh"]) or len(str(row["title_zh"])) > len(str(old["title_zh"])):
            selected[code] = row
    return [_with_compatibility_keys(row) for row in sorted(selected.values(), key=lambda row: str(row["code"]))]


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from an official CNIPA PDF using PyMuPDF."""

    return "\n".join(extract_pdf_pages(pdf_path))


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    """Extract one text string per PDF page for table-of-contents filtering."""

    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("PyMuPDF (fitz) is required to parse CNIPA PDFs") from exc
    with fitz.open(pdf_path) as document:
        return [page.get_text("text") for page in document]


def parse_cnipa_pdf(pdf_path: Path) -> list[dict[str, str | None]]:
    section_match = re.search(r"-([A-H])部", pdf_path.name)
    pages = extract_pdf_pages(pdf_path)
    if section_match:
        section = section_match.group(1)
        actual_start = next(
            (
                i
                for i, page in enumerate(pages)
                if re.search(rf"(?m)^{section}\s*$", page)
                and "目录" not in page[:200]
            ),
            0,
        )
        pages = pages[actual_start:]
    rows = parse_cnipa_text("\n".join(pages))
    if section_match:
        rows = [row for row in rows if str(row["code"]).startswith(section)]
    for row in rows:
        row["source_file"] = pdf_path.name
    return rows


def merge_version_dictionary(
    wipo_scheme_zip: Path,
    wipo_valid_zip: Path,
    cnipa_pdfs: Iterable[Path],
    scheme_version: str,
    wipo_title_list_zip: Path | None = None,
) -> list[dict[str, object]]:
    """Merge WIPO hierarchy/validity with CNIPA Chinese titles for one edition."""

    scheme = parse_wipo_scheme(wipo_scheme_zip)
    valid = parse_wipo_valid_symbols(wipo_valid_zip)
    title_list = {
        row["ipc_code"]: row["title_en"]
        for row in parse_wipo_title_list(wipo_title_list_zip)
    } if wipo_title_list_zip else {}
    zh: dict[str, dict[str, str | None]] = {}
    for pdf_path in cnipa_pdfs:
        for row in parse_cnipa_pdf(pdf_path):
            zh.setdefault(str(row["code"]), row)

    merged: list[dict[str, object]] = []
    for row in scheme:
        code = str(row["code"])
        zh_row = zh.get(code, {})
        merged.append(
            {
                **row,
                "title_en": row.get("title_en") or title_list.get(code),
                "scheme_version": scheme_version,
                "is_valid_symbol": code in valid,
                "title_zh": zh_row.get("title_zh"),
                "title_zh_source": zh_row.get("source_file"),
            }
        )
    return merged
