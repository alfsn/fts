"""Models: fts's model registry, with filters and the Promote/Archive
lifecycle (ported from fts's dashboard, D10)."""

import pandas as pd
import streamlit as st
from workbench import state
from workbench.filters import keep, parameter_filter

STATUSES = ["ALL", "candidate", "production", "archived"]

catalog = state.catalog()
st.title("Model registry")

a, b, c = st.columns(3)
status = a.selectbox("Status", STATUSES)
model_type = b.text_input("Model type (e.g. LSTM)").strip()
instrument = c.text_input("Instrument").strip()

models = catalog.model_repo.list_models(
    model_type=model_type or None,
    instrument_id=instrument or None,
    status=None if status == "ALL" else status,
)
if not models:
    st.info("No models match. The registry is empty until models are trained.")
    st.stop()

params = [{**(m.hyperparameters or {}), **(m.metrics or {})} for m in models]
ranges, text = parameter_filter(params, key="models")
shown = [
    m for m, p in zip(models, params) if keep(p, ranges, text, haystack=m.model_id)
]
if not shown:
    st.warning("No models match the parameter filters.")
    st.stop()

st.dataframe(
    pd.DataFrame(
        [
            {
                "Model ID": m.model_id,
                "Type": m.model_type,
                "Instrument": m.instrument_id,
                "Interval": m.interval,
                "Horizon": m.horizon,
                "Status": m.status,
                "Val IC": (m.metrics or {}).get("val_ic"),
                "Created": m.created_at,
            }
            for m in shown
        ]
    ),
    use_container_width=True,
    column_config={"Val IC": st.column_config.NumberColumn(format="%.4f")},
)

model_id = st.selectbox("Inspect", [m.model_id for m in shown])
detail = catalog.model_repo.get_model_details(model_id)
if detail is None:
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Status", detail.status.upper())
c2.metric("Type", detail.model_type)
c3.metric("Instrument / horizon", f"{detail.instrument_id} (h={detail.horizon})")
c4.metric("ONNX file", "present" if detail.onnx_exists else "missing")

if detail.status != "production":
    if st.button(f"Promote {detail.model_id} to production"):
        if catalog.model_repo.update_model_status(detail.model_id, "production"):
            st.toast(
                "Promoted. The previous production model with the same "
                "type, instrument, interval and horizon is now a candidate."
            )
        else:
            st.error("Model not found.")
        st.rerun()
elif st.button(f"Archive {detail.model_id}"):
    catalog.model_repo.update_model_status(detail.model_id, "archived")
    st.rerun()

left, right = st.columns(2)
left.subheader("Hyperparameters")
left.json(detail.hyperparameters)
right.subheader("Metrics")
right.json(detail.metrics)
