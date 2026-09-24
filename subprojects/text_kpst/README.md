# Chinese Patent KPST Text Subproject

This directory is the new main text-analysis track. It is separate from the
legacy character n-gram/hash-centroid baseline in preprocessing/patent_detail/text_pipeline.py.

## Research target

The baseline follows the method family in Chen et al., Economic Research Journal
(2026), itself based on Kelly, Papanikolaou, Seru and Taddy (KPST):

1. word-level sparse patent vectors;
2. backward inverse document frequency (BIDF);
3. patent-to-patent cosine similarity;
4. five-year backward and forward windows;
5. average rather than summed similarities;
6. same-applicant exclusion;
7. same-IPC backward and all-IPC forward as the baseline comparison design.

The technology-field adjustment is intentionally left as a separate auditable
layer until its exact implementation choices are finalized.

## Deliberate implementation choices

The appendix does not publish a complete tokenizer dictionary, stopword list,
numerical BIDF edge-case convention, or full applicant-disambiguation code.
Those items remain explicit configuration choices.

The first implementation uses Jieba for Chinese word segmentation. This is our
implementation choice, not a claim about the authors' private tokenizer.
A custom patent dictionary and stopword file can be supplied.

Digits are preserved by default because technical strings such as 5G can carry
substantive information.

## Current modules

- src/preprocess.py: Unicode normalization and configurable Jieba tokenization.
- src/tfbidf.py: one-word-one-dimension point-in-time TF-BIDF sparse vectors.
- src/similarity.py: memory-bounded sparse cosine aggregation.
- src/metrics.py: five-year average BS/FS and the unadjusted KPST ratio.

## Integration status

This first refactor does not overwrite existing production outputs. The next
integration step is a shared canonical patent table containing application
metadata, applicant, IPC, abstract and main-claim text, then feeding that table
into this subproject.

The old hash-centroid score is retained only as a legacy baseline. It is not
designated as a robustness test for KPST.
