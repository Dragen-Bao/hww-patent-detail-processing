# Patent text KPST

This subproject adapts the Chen et al. (2026) / KPST patent-text method to a quarterly research design.

Core pipeline:

```
raw CSV
→ canonical patents by application quarter
→ Jieba
→ quarterly TF-BIDF
→ 8-quarter BS / FS
→ same-applicant exclusion
→ quarterly IPC-field adjustment
→ patent quality
→ market-quarter / firm-quarter
```

Main research choices:

- frequency: quarterly throughout the research pipeline;
- comparison sample: granted invention patents;
- text: patent title + abstract;
- BIDF: `log(N_prior / (1 + df_prior))`, using patents from earlier quarters only;
- BS: same IPC main group, previous 8 quarters;
- FS: all IPC, next 8 quarters;
- current quarter is excluded from both BS and FS;
- same applicant: applicant + address;
- field adjustment: recalculated within each application quarter;
- final quality: field adjustment × FS / BS.

The original paper uses a five-year window. The 8-quarter window is the project's quarterly adaptation rather than an exact replication of that choice.

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

- `data/interim/text_kpst/patent_scores/YYYYQn.parquet`
- `data/output/text_kpst/market_quarter_kpst.parquet`
- `data/output/text_kpst/firm_quarter_kpst.parquet`
