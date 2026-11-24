# Data: PhysioNet Apnea-ECG Database v1.0.0

The data is **not committed**. `python scripts/download_data.py` downloads the needed files (~420 MB) and
verifies each one against PhysioNet's `SHA256SUMS.txt`.

| Item | Value |
|---|---|
| Dataset | Apnea-ECG Database v1.0.0, https://physionet.org/content/apnea-ecg/1.0.0/ |
| Licence | Open Data Commons Attribution License v1.0 (ODC-By) |
| Citation | T. Penzel, G.B. Moody, R.G. Mark, A.L. Goldberger, J.H. Peter. "The Apnea-ECG Database." *Computers in Cardiology* 27:255-258 (2000). Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet", *Circulation* 101(23):e215-e220 (2000). |
| Content | 70 overnight single-lead ECG recordings (100 Hz, 16-bit), 7-10 h each |
| Labels | Expert annotation of **every minute**: `A` = apnea or hypopnea present, `N` = normal breathing (based on simultaneously recorded respiration and oxygen saturation) |
| Records | a01-a20, b01-b05, c01-c10 (learning set), x01-x35 (test set; its labels were released after the challenge, in `x??.apn` / `test-dataset-annos.txt`) |
| Extra | `additional-information.txt`: per-recording apnea minutes, AHI, age, sex, height, weight |

Files used per record: `.dat`/`.hea` (ECG), `.apn` (per-minute labels), `.qrs` (QRS annotations distributed with the database).

## Clinical meaning and scope
- Obstructive sleep apnea (OSA) is a **sleep-related breathing disorder** (upper-airway obstruction during sleep). It is respiratory, but it is **not** a disease of the lung tissue.
- The labels come from respiratory signals. The ECG is used only as the *input*, because apnea leaves well-documented traces in it: cyclic variation of heart rate and breathing-induced R-wave amplitude modulation (ECG-derived respiration).
- See the main README, section "Relation to the resume wording", for why this dataset was chosen over others.

## Subjects and leakage control
Subject IDs are not published, but several recordings share identical age, sex, height and weight (e.g. a01/a14),
meaning the same person was recorded on different nights. Records are grouped into pseudo-subjects on these four
attributes, and whole subjects are assigned to train, validation or test. Grouping on identical attributes is conservative: it can only merge people, never split one person.
