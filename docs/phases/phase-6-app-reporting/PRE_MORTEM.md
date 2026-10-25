# Phase 6 Pre-Mortem: What Could Go Wrong & Mitigations

**Phase:** Phase 6 (Streamlit App & Reporting - Tier P0)  
**Date:** 2026-10-25  

---

## 1. Potential Failure Modes

### Risk 1: Streamlit Session State Invalidation or Rerun Latency
- **Failure:** Every widget interaction triggering complete model retraining and pipeline execution causes painful UI freezes.
- **Mitigation:**
  - Cache heavy computations using `@st.cache_data` and `@st.cache_resource`.
  - Load pre-trained models or pre-computed benchmark tables when exploring default views.
  - Store user selections and simulation logs in `st.session_state`.

### Risk 2: Headless CI / Pytest Failure on Streamlit App
- **Failure:** Streamlit apps without a browser might fail when executed under standard pytest.
- **Mitigation:**
  - Use `streamlit.testing.v1.AppTest.from_file()` to run programmatic headless smoke tests on each page.
  - Assert that `app.exception` is empty and critical title elements render.

### Risk 3: SQLite Concurrency or File Lock Conflicts on Windows
- **Failure:** SQLite database locked if multiple threads or processes access `runs.db` simultaneously.
- **Mitigation:**
  - Use short connection lifetimes (`with sqlite3.connect(...) as conn:`) with `timeout=10.0` and `WAL` journal mode.
