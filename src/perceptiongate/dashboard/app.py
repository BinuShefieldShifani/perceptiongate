"""PerceptionGate dashboard.

pip install -e ".[dashboard]"
streamlit run src/perceptiongate/dashboard/app.py -- --runs ~/perceptiongate-runs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import streamlit as st

from perceptiongate.dashboard.data import baseline, discover, overview, vru_heatmap

p = argparse.ArgumentParser()
p.add_argument("--runs", type=Path, default=Path.home() / "perceptiongate-runs")
p.add_argument("--baseline", type=Path, default=Path("baselines/bdd100k-v1"))
args, _ = p.parse_known_args()

st.set_page_config(page_title="PerceptionGate", layout="wide")
st.title("PerceptionGate")
st.caption(f"Runs under `{args.runs}` · baseline `{args.baseline}`")

runs = discover(args.runs)
base = baseline(args.baseline) if args.baseline.is_dir() else None
if not runs:
    st.warning("No runs found. Unzip an Actions artifact (kaggle-<name>.zip) into the runs folder.")
    st.stop()

st.subheader("All runs")
st.dataframe(pd.DataFrame(overview(runs, base)), use_container_width=True, hide_index=True)

st.subheader("VRU recall by slice")
heat = pd.DataFrame(vru_heatmap(runs, base)).T
st.dataframe(heat.style.background_gradient(cmap="RdYlGn", vmin=0.2, vmax=0.6).format("{:.3f}"),
             use_container_width=True)  # fmt: skip

st.subheader("Run details")
choice = st.selectbox("Run", [r["run"] for r in runs])
run = next(r for r in runs if r["run"] == choice)
c1, c2, c3 = st.columns(3)
c1.metric("Decision", run["decision"])
c2.metric("VRU recall", run["VRU recall"], f"{run['Δ VRU recall (pt)']:+.1f} pt"
          if run["Δ VRU recall (pt)"] is not None else None)  # fmt: skip
c3.metric("Commit", run["commit"] or "unknown")

tab_gate, tab_fail, tab_agent = st.tabs(["Gate", "Failure mining", "Agent evaluation"])
with tab_gate:
    st.markdown(run["gate_md"])
with tab_fail:
    f = run["failure"]
    if f:
        st.write(f"Misses {f['candidate']['missed']} of {f['candidate']['boxes']} VRU boxes.")
        st.dataframe(pd.DataFrame(f["candidate"]["by_size"]), hide_index=True)
        for h in f["headline"]:
            st.write(f"- {h}")
    else:
        st.info("No failure report for this run (runs before step 9 don't have one).")
with tab_agent:
    a = run["agent_eval"]
    if a:
        st.write(f"`{a['model']}`: grounded first try {a['grounded_first_try']:.0%}, LLM text used "
                 f"{a['used_llm_text']:.0%}, cause named {a['cause_named']:.0%}")  # fmt: skip
        st.dataframe(pd.DataFrame(a["rows"]), hide_index=True)
    else:
        st.info("No agent evaluation for this run (runs before step 10 don't have one).")
