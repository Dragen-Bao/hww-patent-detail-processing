# Chinese Patent KPST Text Pipeline

This is the main patent-text quality track. The old character n-gram/hash-centroid code in `preprocessing/patent_detail/text_pipeline.py` is retained as legacy code only.

## Source-grounded baseline

The supplied Chen et al. (2026) appendix defines the core procedure as:

1. real word-level one-hot vectors;
2. TF × BIDF, with `BIDF_pw = log(N_prior / (1 + df_prior,w))`;
3. L2 normalization and cosine similarity;
4. five application years backward and forward;
5. similarity sums divided by the patent counts in the corresponding year window;
6. same-applicant exclusion using applicant + address;
7. Appendix Table 1 method 6 as the baseline: backward similarity within the same IPC field, forward similarity across all IPC fields, with same-applicant exclusion;
8. annual technology-field adjustment from relative backward similarity, multiplied by FS/BS.

The appendix defines `N_k` as the number of granted invention patents applied for in year `k`. In the local database, some granted inventions can have no usable configured text. The executable comparison universe is therefore defined explicitly as granted inventions with a non-empty vector. The pipeline writes `text_coverage_by_year.csv` so the gap from all granted inventions is visible rather than silently changing the denominator.

## Explicit implementation choices

The appendix does not disclose every engineering detail. These are transparent project choices:

- Chinese segmentation: Jieba.
- Default text: patent title + abstract.
- Main claim is retained in the canonical patent layer but not included by default.
- No stopword list and no frequency pruning by default; BIDF handles common words.
- BIDF history is strict-prior by application date with same-day batch freeze.
- IPC field key is the main group parsed from `IPC主分类号`, e.g. `H01M4/02 -> H01M4`.

Appendix Table 1 and the surrounding paragraph identify method 6 as forward all-IPC + backward same-IPC + same-applicant exclusion. One nearby footnote is textually inconsistent with that table; the implementation follows the explicit method-6 definition in the table and surrounding paragraph.

## Structure

- `preprocessing/patent_detail/canonical.py`: reads the actual 31-column raw CSV, repairs wrapped rows, deduplicates by application number, keeps title/abstract/main claim, builds applicant identity, parses IPC main group, and writes reusable patent/firm-edge Parquet partitions.
- `subprojects/text_kpst/src/preprocess.py`: Jieba word segmentation.
- `subprojects/text_kpst/src/tfbidf.py`: equations (1)-(3), strict-prior TF-BIDF, L2 normalization.
- `subprojects/text_kpst/src/scalable.py`: year-sharded sparse matrices and memory-bounded BS/FS scoring. It uses the exact identity `sum cosine = row dot vector-sum`, so it does not build a full N×N pair matrix.
- `subprojects/text_kpst/src/metrics.py`: equations (7)-(10), including field adjustment.
- `subprojects/text_kpst/src/aggregation.py`: patent-level to market-quarter and listed-company-quarter outputs.

## Run

Install dependencies:

```bash
uv sync
```

Run the full pipeline:

```bash
python -m subprojects.text_kpst.run --config subprojects/text_kpst/config/default.json
```

Or resume by stage:

```bash
python -m subprojects.text_kpst.run --stage canonical
python -m subprojects.text_kpst.run --stage vectors
python -m subprojects.text_kpst.run --stage score
python -m subprojects.text_kpst.run --stage aggregate
```

Main outputs:

- `patent_scores/kpst-YYYY.parquet`: patent-level BS, FS, field adjustment and quality.
- `firm_quarter_kpst.parquet/csv`: listed-company-quarter panel.
- `market_quarter_kpst.parquet/csv`: market-quarter panel.
- `run_manifest.json`: configuration and method notes.

The final five years do not have a complete five-year forward window. They are retained with `forward_window_complete=0`, and `kpst_quality_adjusted_complete` is missing unless both backward and forward windows are complete.


## Data-integrity safeguards

- Missing applicant identity does not trigger same-applicant exclusion. Empty identities are never grouped together; the rule is symmetric for backward and forward comparisons.
- `text_coverage_by_year.csv` reports all granted inventions, tokenized patents, vectorizable patents, missing applicant identity/address, and missing IPC domain by year.
- Every actual rebuild of the canonical, vector, or score stage clears that stage's previous shard directories before writing. This prevents stale Parquet/NPZ files from contaminating a forced rebuild or a rebuild after parameter/input changes.
