# Patent text KPST

This subproject implements the Chen et al. (2026) adaptation of KPST for the local Chinese patent data.

Core pipeline:

```
raw CSV
→ canonical patents
→ Jieba
→ TF-BIDF
→ 5-year BS / FS
→ same-applicant exclusion
→ IPC-field adjustment
→ patent quality
→ market-quarter / firm-quarter
```

Method choices:

- comparison sample: granted invention patents;
- text: patent title + abstract;
- BIDF: `log(N_prior / (1 + df_prior))`;
- backward: same IPC main group, previous 5 years;
- forward: all IPC, next 5 years;
- same applicant: applicant + address;
- final quality: field adjustment × FS / BS.

Only three data-handling rules are treated as essential infrastructure: repair the malformed wrapped CSV, deduplicate by application number, and check that sparse-matrix rows match metadata rows.

Install and run:

```bash
uv sync
python -m subprojects.text_kpst.run
```

For a large dataset, stages can be run separately:

```bash
python -m subprojects.text_kpst.run --stage canonical
python -m subprojects.text_kpst.run --stage vectors
python -m subprojects.text_kpst.run --stage score
python -m subprojects.text_kpst.run --stage aggregate
```

Outputs:

- `data/interim/text_kpst/patent_scores/YYYY.parquet`
- `data/output/text_kpst/market_quarter_kpst.parquet`
- `data/output/text_kpst/firm_quarter_kpst.parquet`
