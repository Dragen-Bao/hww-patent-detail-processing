"""Official IPC source specifications and download helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any
from urllib.request import Request, urlopen


IPC_VERSION = "2026.01"
HISTORICAL_REFERENCE_VERSION = "2025.01"
WIPO_ROOT = "https://www.wipo.int/classifications/data/ipc/ITSupport_and_download_area/20260101"
WIPO_ROOT_2025 = "https://www.wipo.int/classifications/data/ipc/ITSupport_and_download_area/20250101"
CNIPA_ROOT = "https://www.cnipa.gov.cn/module/download/downfile.jsp?classid=0&filename="


SOURCE_SPECS = [
    {
        "name": "wipo_scheme",
        "url": f"{WIPO_ROOT}/MasterFiles/ipc_scheme_20260101.zip",
        "filename": "ipc_scheme_20260101.zip",
        "source_language": "en",
    },
    {
        "name": "wipo_valid_symbols",
        "url": f"{WIPO_ROOT}/valid_symbol_list/ipc_valid_symbols_20260101.zip",
        "filename": "ipc_valid_symbols_20260101.zip",
        "source_language": "en",
    },
    {
        "name": "wipo_concordance",
        "url": f"{WIPO_ROOT}/MasterFiles/ipc_concordancelist_20260101.zip",
        "filename": "ipc_concordancelist_20260101.zip",
        "source_language": "en",
    },
    {
        "name": "wipo_title_list",
        "url": f"{WIPO_ROOT}/IPC_scheme_title_list/EN_ipc_title_list_20260101.zip",
        "filename": "EN_ipc_title_list_20260101.zip",
        "source_language": "en",
    },
    {
        "name": "wipo_2025_scheme",
        "url": f"{WIPO_ROOT_2025}/MasterFiles/ipc_scheme_20250101.zip",
        "filename": "ipc_scheme_20250101.zip",
        "source_language": "en",
    },
    {
        "name": "wipo_2025_valid_symbols",
        "url": f"{WIPO_ROOT_2025}/valid_symbol_list/ipc_valid_symbols_20250101.zip",
        "filename": "ipc_valid_symbols_20250101.zip",
        "source_language": "en",
    },
    {
        "name": "wipo_2025_title_list",
        "url": f"{WIPO_ROOT_2025}/IPC_scheme_title_list/EN_ipc_title_list_20250101.zip",
        "filename": "EN_ipc_title_list_20250101.zip",
        "source_language": "en",
    },
    {
        "name": "cnipa_2025_A",
        "url": CNIPA_ROOT + "490dd33a669c43d7b2dccaabaa730ce8.pdf&showname=2025.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-A%E9%83%A8.pdf",
        "filename": "2025.01版IPC分类表-A部.pdf",
        "source_language": "zh",
    },
    {
        "name": "cnipa_2025_B",
        "url": CNIPA_ROOT + "e75c01f9db70483aad983ea8a99aa2a1.pdf&showname=2025.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-B%E9%83%A8.pdf",
        "filename": "2025.01版IPC分类表-B部.pdf",
        "source_language": "zh",
    },
    {
        "name": "cnipa_2025_C",
        "url": CNIPA_ROOT + "a8b47d2d872742a7ab634892159ad6aa.pdf&showname=2025.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-C%E9%83%A8.pdf",
        "filename": "2025.01版IPC分类表-C部.pdf",
        "source_language": "zh",
    },
    {
        "name": "cnipa_2025_D",
        "url": CNIPA_ROOT + "84c5c0a878724a5099241142f32de442.pdf&showname=2025.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-D%E9%83%A8.pdf",
        "filename": "2025.01版IPC分类表-D部.pdf",
        "source_language": "zh",
    },
    {
        "name": "cnipa_2025_E",
        "url": CNIPA_ROOT + "eedad1fd7ceb4e5d87446b969a090f48.pdf&showname=2025.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-E%E9%83%A8.pdf",
        "filename": "2025.01版IPC分类表-E部.pdf",
        "source_language": "zh",
    },
    {
        "name": "cnipa_2025_F",
        "url": CNIPA_ROOT + "49e52fa29aad437fa859d3939ce64d9b.pdf&showname=2025.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-F%E9%83%A8.pdf",
        "filename": "2025.01版IPC分类表-F部.pdf",
        "source_language": "zh",
    },
    {
        "name": "cnipa_2025_G",
        "url": CNIPA_ROOT + "a46abe466a524434bd2d00dc6d4e3fa7.pdf&showname=2025.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-G%E9%83%A8.pdf",
        "filename": "2025.01版IPC分类表-G部.pdf",
        "source_language": "zh",
    },
    {
        "name": "cnipa_A",
        "url": CNIPA_ROOT + "84f50032f9ce4301ab100f5c403e000b.pdf&showname=2026.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-A%E9%83%A8.pdf",
        "filename": "2026.01版IPC分类表-A部.pdf",
        "source_language": "zh",
    },
    {
        "name": "cnipa_B",
        "url": CNIPA_ROOT + "a4f808817050429e849bd818907cfc1d.pdf&showname=2026.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-B%E9%83%A8.pdf",
        "filename": "2026.01版IPC分类表-B部.pdf",
        "source_language": "zh",
    },
    {
        "name": "cnipa_C",
        "url": CNIPA_ROOT + "99c9d77c96ae4000ab1ec35c75a95881.pdf&showname=2026.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-C%E9%83%A8.pdf",
        "filename": "2026.01版IPC分类表-C部.pdf",
        "source_language": "zh",
    },
    {
        "name": "cnipa_D",
        "url": CNIPA_ROOT + "7719d52ee72b4cedb8a925a86d8932cb.pdf&showname=2026.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-D%E9%83%A8.pdf",
        "filename": "2026.01版IPC分类表-D部.pdf",
        "source_language": "zh",
    },
    {
        "name": "cnipa_E",
        "url": CNIPA_ROOT + "b5822d0c2cbc4daa823e95ebb9d70cd2.pdf&showname=2026.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-E%E9%83%A8.pdf",
        "filename": "2026.01版IPC分类表-E部.pdf",
        "source_language": "zh",
    },
    {
        "name": "cnipa_F",
        "url": CNIPA_ROOT + "826c947f24cd469485b8c75e76b3226e.pdf&showname=2026.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-F%E9%83%A8.pdf",
        "filename": "2026.01版IPC分类表-F部.pdf",
        "source_language": "zh",
    },
    {
        "name": "cnipa_G",
        "url": CNIPA_ROOT + "61ec0e15ff694bbc98f302ab0e4438ee.pdf&showname=2026.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-G%E9%83%A8.pdf",
        "filename": "2026.01版IPC分类表-G部.pdf",
        "source_language": "zh",
    },
    {
        "name": "cnipa_H",
        "url": CNIPA_ROOT + "e68dc3300ca44d2d9e606a6a5c979725.pdf&showname=2026.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-H%E9%83%A8.pdf",
        "filename": "2026.01版IPC分类表-H部.pdf",
        "source_language": "zh",
    },
    {
        "name": "cnipa_2025_H",
        "url": CNIPA_ROOT + "9ec917cb185443e9bfa8848adda024a9.pdf&showname=2025.01%E7%89%88IPC%E5%88%86%E7%B1%BB%E8%A1%A8-H%E9%83%A8.pdf",
        "filename": "2025.01版IPC分类表-H部.pdf",
        "source_language": "zh",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_one(spec: dict[str, str], output_dir: Path, timeout: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / spec["filename"]
    source_version = "2025.01" if "2025" in spec["name"] or "2025" in spec["filename"] else "2026.01"
    if target.exists():
        return {
            **spec,
            "source_version": source_version,
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "downloaded_at_utc": None,
            "reused_existing": True,
        }

    request = Request(spec["url"], headers={"User-Agent": "IPC-dictionary-builder/1.0"})
    with urlopen(request, timeout=timeout) as response:
        with tempfile.NamedTemporaryFile(prefix=target.name, suffix=".part", dir=output_dir, delete=False) as temp:
            partial = Path(temp.name)
            while chunk := response.read(1024 * 1024):
                temp.write(chunk)
        partial.replace(target)
        content_type = response.headers.get("Content-Type", "")

    return {
        **spec,
        "source_version": source_version,
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "content_type": content_type,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "reused_existing": False,
    }


def download_sources(output_dir: Path, timeout: int = 60) -> dict[str, Any]:
    records = [_download_one(spec, output_dir, timeout) for spec in SOURCE_SPECS]
    manifest = {
        "ipc_version": IPC_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": records,
    }
    manifest_path = output_dir / "ipc_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"manifest": manifest_path, "sources": records}
