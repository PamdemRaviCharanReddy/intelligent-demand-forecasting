# Phase 0 Brief: Discovery, Scope & Architecture

## 1. Goal
Establish a scientifically rigorous, fully reviewable architectural foundation, data contracts, evaluation protocol, and runnable code skeleton for the Intelligent Demand Forecasting System (IDFS). Phase 0 produces no trained ML models or user interfaces; it locks architectural decisions through formal Architecture Decision Records (ADRs), constructs testable interfaces and schemas, and defines the roadmap for subsequent phases.

## 2. Inputs Available
- **Source Material**: Student mid-term report, presentation slides (PPT), introductory draft, and simple pipeline diagram (summarized in `docs/source/PROJECT_SOURCE_DIGEST.md`).
- **Academic Context**: Special Project at ICFAI Foundation for Higher Education (IcfaiTech), Hyderabad. Mentored by Dr. Sukanta Das and Dr. Sandeep Kumar Panda `[VERIFY]`.
- **Environment Findings**: Windows 11 host with AMD 8-core CPU, 12GB RAM, NVIDIA RTX 3050 Laptop GPU (4GB VRAM), PortableGit, and Python 3.12 managed via `uv` (`docs/ENVIRONMENT.md`).

## 3. Constraints & Operating Environment
- **Hardware Limitations**: Target models must train comfortably within 12GB RAM and CPU/GPU memory caps without crashing or causing excessive swap.
- **Data Availability Reality**: Kaggle API access requires student credentials and competition agreement; default pipeline must execute deterministically on an authentic synthetic data generator.
- **Academic Viva Requirements**: All modules must be explainable in non-trivial technical depth to examiners (defending against leakage, metric gaming, or ungrounded claims).

## 4. Key Unknowns & Risks Addressed
- *Inconsistency in Student Sources*: Variations in model lists, metrics, and mentor names across reports and slides are formally reconciled in `docs/DOCUMENT_FIXES.md`.
- *Leakage Vulnerabilities*: Rolling statistics and lag features must never see future rows (`.shift(1)` rule enforced by construction).
- *Inventory Modeling Disconnect*: Standard forecasting metrics (MAE, RMSE) fail to reflect true stockout vs. holding cost trade-offs; N1 and N2 address this directly.
