# Golden Test — Team-Leader Approval and Freeze Record

**Project:** Arabic PII De-identification — NLP Applications, KAUST Academy 2026  
**Topic:** 6, Option A  
**Version:** 1.0 (approved release)  
**Approval status:** **APPROVED BY TEAM LEADER — 17 November 2026**  
**GitHub freeze evidence:** This record is committed together with the approved golden file. The dedicated commit on `main` (visible in GitHub history) is the evidence of remote freeze; the dataset was approved on 17 November 2026 before planned model training.  
**Approved dataset path:** `data/golden_test.jsonl`

## 1. Approval authority and provenance

On 17 November 2026, the team leader, **Abdulrahman Alharthi**, expressly authorized final approval of the reviewed 100-example golden set and the recorded `g090` annotation correction. The leader reported that each of four team members authored/annotated 25 examples and peer-reviewed 25 examples, with reciprocal review pairs. The leader also reported that this arrangement had been explained to the instructor. This record documents the leader's account and approval; it is **not an independent verification** of each participant's activity or of instructor acknowledgement.

The original source is `golden_test_draft.jsonl` (100 JSONL rows). The original draft, the 10-row template, and model code were not altered in preparing this release. The approved version updates `written_by` and `checked_by` from `TBD` according to the leader-confirmed author/reviewer pairing, and adds one named-organization entity in `g090`. No sentence text, row ID, dialect, category, or other entity annotation changed.

## 2. Authorship and checking matrix

| Member | Rows authored | Rows checked | Checker of this member's rows |
|---|---:|---:|---|
| Abdulrahman Alharthi | 25 | 25 | Yasser Al-Hajlah |
| Yasser Al-Hajlah | 25 | 25 | Abdulrahman Alharthi |
| Musaad Al-Qahtani | 25 | 25 | Mohammed Al-Rashidi |
| Mohammed Al-Rashidi | 25 | 25 | Musaad Al-Qahtani |

In all 100 rows, the author differs from the checker. The per-row allocation and basis of approval are in `docs/golden_test_review_assignments.csv`. `APPROVED_BY_TEAM_LEADER` is an administrative confirmation by the leader, **not** a claim that a third party re-verified every review action.

## 3. Approved annotation change

**`g090`:** Add `ORG: عيادة الأمل` to the sentence `راجعت ميس الشمري عيادة الأمل، ثم اتصلت على +966 50 000 0013.` A specifically named clinic is an organization under `docs/annotation_guidelines.md`. The pre-existing `PERS` and `PHONE` labels are retained. The sentence itself is unchanged. The leader explicitly approved this correction. This is the **only** difference in entity labels compared with the attached draft. The evidence is preserved in `docs/golden_test_annotation_changes.csv`.

The existing annotations in `g030` (attached lām in `للقصيم`), `g089` (10-digit non-ID serial as a negative control) and `g097` (metalinguistic `ريم` versus the named person `ريم الحارثي`) were kept unchanged under the existing annotation policy.

## 4. Dataset profile and validation

- **100** rows, unique IDs `g001`–`g100`, and unique source sentences.
- **20** negative-control rows (no entity).
- **35** Saudi-dialect rows.
- **121** entity spans: `PERS` 19, `LOC` 15, `ORG` 18, `DATE` 14, `NATIONAL_ID` 13, `PHONE` 15, `MRN` 13, `EMAIL` 14.
- Ran the project's original strict validator **without** `--allow-unreviewed` using the approved dataset; **PASSED, 0 errors, 0 warnings**.
- Bounded lexical overlap screening using the existing `GoldenIndex` logic found **0 detected overlaps** against 4 repository few-shot prompt examples and 600 synthetic examples generated with `generate_synthetic(600, 13)` in the examined snapshot.

**Limitations:** validator success cannot independently prove annotation semantics, human participation or absence of real PII. The overlap screening does not cover any separate/future training corpus or paraphrase-based leakage. Before training, screen **all actual training/dev data** against this frozen set, and verify dataset licences and synthetic-data provenance. Do not edit the golden test in response to model scores.

## 5. Content integrity (SHA-256)

- Original draft `golden_test_draft.jsonl`: `dd1dc2a249ace24e4f3e0a9a5a5ef5d261054ed31f6457a1b88f27e1211b94d0`
- Approved `data/golden_test.jsonl`: `9297f0d21c44ad7f7711af3c254a1e5ff9da1cf863e93162386151993d590459`

These hashes identify the exact reviewed source and proposed frozen bytes. They do not prove remote publication.

## 6. Freeze and post-freeze change policy

- [x] Team leader confirms per-row author/reviewer pairings and 25/25 allocation.
- [x] Team leader approves the only entity-label correction (`g090`).
- [x] Strict validator passes with 0 errors and 0 warnings.
- [x] Existing prompt examples and seeded 600-example dataset screened (scope limited as above).
- [x] **Release prepared for GitHub freeze commit.** Verify the dedicated commit is present on `origin/main`. Include the five approved files listed in this release in **one dedicated commit** on `main`, then archive the full commit SHA/URL as proof of freeze before training.

No additional fine-tuning or evaluation on the golden test should start before that remote commit. If a genuine annotation error is discovered after freeze, document it explicitly in a new commit, preserve the original commit, and redo the affected evaluation; never silently revise the approved dataset.