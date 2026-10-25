# Items Requiring Human Attention

This file tracks questions, credentials, or manual setup tasks that require human input. In accordance with Operating Principle 8, the autonomous pipeline does **not** halt for these items; it uses documented sensible defaults and proceeds with all unblocked tasks.

---

## 1. Credentials & Data Access

### A. Kaggle API Credentials (for real Favorita store sales dataset)
- **Status**: Not found (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables are absent).
- **Default Action Taken**: System proceeds autonomously using the fully calibrated synthetic data generator (`src/idfs/data/synthetic.py`).
- **Instructions for Students**:
  1. Log into your Kaggle account at [kaggle.com](https://www.kaggle.com).
  2. Navigate to your Account Settings -> **API** -> Click **Create New Token**.
  3. This downloads `kaggle.json`. Place this file at `C:\Users\PAMDEM RAVI CHARAN\.kaggle\kaggle.json` (or set the environment variables in `.env`).
  4. Accept the competition rules for [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules).
  5. Once completed, Phase 1 can ingest and process the real competition data.

---

## 2. Academic & Metadata Confirmations

### B. Project Mentor Confirmation
- **Status**: The mid-term report lists Dr. Sukanta Das; the PPT lists Dr. Sukanta Das and Dr. Sandeep Kumar Panda.
- **Default Action Taken**: Both faculty members are acknowledged in documentation marked `[VERIFY]`.
- **Question for Students**: Should the final report acknowledge Dr. Sukanta Das only or both Dr. Sukanta Das and Dr. Sandeep Kumar Panda?

### C. Student Roll Number Clarification
- **Status**: Student Jaiti Vijaya Shravani is listed as `24STUCHH01010756` (8 digits) while all other team members have 6 digits (e.g., `24STUCHH010749`).
- **Default Action Taken**: Flagged with `[VERIFY]` as likely `24STUCHH010756`.
- **Question for Students**: Please confirm the exact roll number.

### D. Project Hard Deadline
- **Status**: Not specified.
- **Default Action Taken**: Stored as `deadline: null` in `configs/default.yaml`. Planned in relative milestones (W1 to W8).
- **Question for Students**: If a hard submission date or viva presentation date is scheduled, please provide it.

---

## 3. Tooling Environment (GitHub CLI)
- **Status**: `gh` CLI is not installed on this machine.
- **Default Action Taken**: Git operations are performed locally on branches and merged to `main`. Remote PR creation and GitHub issues can be synced once `gh` is authenticated or pushed directly via `git push origin main`.
