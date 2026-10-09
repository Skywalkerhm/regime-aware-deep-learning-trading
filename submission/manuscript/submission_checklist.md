# PLOS ONE Submission Checklist

Author: Min Huang — The Chinese University of Hong Kong, Shenzhen — minhuang1@link.cuhk.edu.cn
— ORCID 0009-0007-4991-3221. `[x]` = already satisfied by the current draft/package.

## Style and format (PLOS ONE)
- [x] Manuscript supplied as DOCX (available in Markdown source).
- [x] Standard font; headings limited to three levels.
- [x] Double-spaced, single column.
- [x] Continuous line numbers, page numbers.
- [x] No footnotes.
- [x] English; abbreviations defined at first use.

## Manuscript organization
- [x] Title page (title, short title, author, affiliation, corresponding author, ORCID).
- [x] Abstract <= 300 words, unstructured, no citations (288 words).
- [x] Introduction (background, problem, brief literature, aim).
- [x] Materials and Methods (including a Statistical analysis subsection).
- [x] Results; Discussion.
- [x] Acknowledgments (non-author contributors only; no funding sources).
- [x] References (Vancouver, numbered, cited in order; 25 entries).
- [x] Supporting information captions listed at the end of the manuscript.

## Figures and tables
- [x] Figures are separate files, cited in ascending order (`Fig1.png`, `Fig2.png`, `Fig3.png`).
- [x] Figure captions inserted immediately after the first citing paragraph.
- [x] Tables placed in the manuscript, cited in ascending order (Table 1-7), label + title above.
- [ ] Optional: export figures as 300-dpi TIFF if the submission system requests it.

## Required statements (see `statements.md`)
- [x] Data Availability Statement (Supporting Information + public GitHub release).
- [x] Code Availability Statement (MIT).
- [x] Funding Statement ("The author received no specific funding for this work.").
- [x] Competing Interests Statement ("none").
- [x] Author Contributions (CRediT; sole author, all roles).
- [x] Ethics Statement (public data only).

## Data and code
- [x] Minimal data set (derived bundle, 19 columns) included as S1 Data; archived by PLOS in figshare.
- [x] No raw third-party price/on-chain series redistributed; sources cited.
- [x] Analysis code provided (S1 Code + GitHub).
- [x] One-command reproduction (`reproduce/run_all.sh`); SHA-256 manifest; verification report.
- [x] All experiments reproduce bit-identically; analysis test suite passes.

## Author actions before submitting
- [ ] Add a `LICENSE` file (MIT) to the GitHub repository.
- [ ] Push the `reproduce/` workspace to the repository; optionally tag `v1.0`.
- [ ] Optionally name 1-3 PLOS ONE Academic Editors and any opposed reviewers in the cover letter.
- [ ] Confirm full author lists / DOIs for the preprint references at camera-ready.
- [ ] Enter author contributions and funding in the submission system.
