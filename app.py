# app.py
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Viral Architect Dashboard", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def normalize_row(row, eps=1e-12):
    s = float(np.sum(row))
    if abs(s) < eps:
        return row
    return row / s

def is_absorbing_row(row, i, tol=1e-9):
    # absorbing if P[i,i] == 1 and all other entries == 0 (within tol)
    for j, v in enumerate(row):
        if j == i:
            if abs(v - 1.0) > tol:
                return False
        else:
            if abs(v) > tol:
                return False
    return True

def partition_states(P, state_names, tol=1e-9):
    absorbing_idx = [i for i in range(P.shape[0]) if is_absorbing_row(P[i, :], i, tol=tol)]
    transient_idx = [i for i in range(P.shape[0]) if i not in absorbing_idx]
    return transient_idx, absorbing_idx

def fundamental_and_absorption(P, transient_idx, absorbing_idx):
    # Q = transient->transient, R = transient->absorbing
    Q = P[np.ix_(transient_idx, transient_idx)]
    R = P[np.ix_(transient_idx, absorbing_idx)]
    I = np.eye(Q.shape[0])
    # F = (I - Q)^-1
    F = np.linalg.inv(I - Q)
    # B = F * R
    B = F @ R
    return Q, R, F, B

def coerce_to_row_stochastic(P):
    P2 = P.copy().astype(float)
    for i in range(P2.shape[0]):
        P2[i, :] = np.clip(P2[i, :], 0.0, 1.0)
        P2[i, :] = normalize_row(P2[i, :])
    return P2

def what_if_reduce_newbie_dropout(P, state_names, delta, dropout_state_name="Deleted Account"):
    """
    Reduce Newbie->Deleted by delta, then reassign that probability to Newbie->Newbie (stay) by default.
    Keeps row-stochastic if possible and clamps at [0,1].
    """
    P2 = P.copy().astype(float)
    s_newbie = state_names.index("Newbie")
    s_deleted = state_names.index(dropout_state_name)

    current_dropout = P2[s_newbie, s_deleted]
    applied = min(max(delta, 0.0), current_dropout)  # can't reduce more than current dropout

    # Move probability mass from dropout to "stay" (Newbie->Newbie)
    P2[s_newbie, s_deleted] = current_dropout - applied
    P2[s_newbie, s_newbie] = P2[s_newbie, s_newbie] + applied

    # Clamp and renormalize just in case of numerical weirdness
    P2[s_newbie, :] = np.clip(P2[s_newbie, :], 0.0, 1.0)
    P2[s_newbie, :] = normalize_row(P2[s_newbie, :])
    return P2, applied

# -----------------------------
# Default 6-state model from prompt
# -----------------------------
state_names = [
    "Newbie",
    "Casual",
    "Power User",
    "Community Leader",
    "Verified Legend",
    "Deleted Account",
]

# Build a default P consistent with Part 2 description:
# S1: 0.40 stay, 0.20 -> S2, 0.40 drop (S6)
# S2: 0.30 stay, 0.30 -> S3, 0.40 drop
# S3: 0.30 stay, 0.50 -> S4, 0.20 drop
# S4: 0.30 stay, 0.60 -> S5, 0.10 drop
# S5 absorbing, S6 absorbing
P_default = np.array([
    [0.40, 0.20, 0.00, 0.00, 0.00, 0.40],
    [0.00, 0.30, 0.30, 0.00, 0.00, 0.40],
    [0.00, 0.00, 0.30, 0.50, 0.00, 0.20],
    [0.00, 0.00, 0.00, 0.30, 0.60, 0.10],
    [0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
], dtype=float)

# -----------------------------
# UI
# -----------------------------
st.title("The Viral Architect — Absorbing Markov Chain Dashboard")
st.write(
    "Enter or edit the transition matrix \(P\). The app will identify transient/absorbing states, "
    "compute \(Q, R, F=(I-Q)^{-1}\), and \(B=FR\), and show the probability a Newbie becomes a Verified Legend."
)

left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("1) Input: Transition matrix P (6×6)")
    st.caption("Rows should sum to 1. Values will be clipped to [0,1]. You can also toggle auto-normalization.")
    auto_norm = st.toggle("Auto-normalize each row to sum to 1", value=True)

    P_df = pd.DataFrame(P_default, index=state_names, columns=state_names)
    edited = st.data_editor(
        P_df,
        use_container_width=True,
        num_rows="fixed",
        key="p_editor",
    )

    P = edited.to_numpy(dtype=float)
    if auto_norm:
        P = coerce_to_row_stochastic(P)

    st.caption("Current row sums:")
    row_sums = pd.Series(P.sum(axis=1), index=state_names, name="Row sum")
    st.dataframe(row_sums.to_frame(), use_container_width=True)

with right:
    st.subheader("2) What-If: Reduce Newbie dropout")
    what_if_on = st.toggle("Enable What-If scenario", value=False)
    delta = st.slider(
        "Reduce P(Newbie → Deleted) by Δ",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
        disabled=not what_if_on,
    )
    st.caption("Δ is reallocated to P(Newbie → Newbie) (i.e., improved retention in the first state).")

# -----------------------------
# Compute baseline + what-if
# -----------------------------
def compute_outputs(P_matrix):
    transient_idx, absorbing_idx = partition_states(P_matrix, state_names)
    Q, R, F, B = fundamental_and_absorption(P_matrix, transient_idx, absorbing_idx)

    transient_names = [state_names[i] for i in transient_idx]
    absorbing_names = [state_names[i] for i in absorbing_idx]

    Q_df = pd.DataFrame(Q, index=transient_names, columns=transient_names)
    R_df = pd.DataFrame(R, index=transient_names, columns=absorbing_names)
    F_df = pd.DataFrame(F, index=transient_names, columns=transient_names)
    B_df = pd.DataFrame(B, index=transient_names, columns=absorbing_names)

    # Success metric: P(Newbie -> Verified Legend), if those states exist in B
    success = np.nan
    if "Newbie" in transient_names and "Verified Legend" in absorbing_names:
        success = float(B_df.loc["Newbie", "Verified Legend"])

    # Life expectancy tau for Newbie: sum of row(Newbie) in F
    tau = np.nan
    if "Newbie" in transient_names:
        tau = float(F_df.loc["Newbie", :].sum())

    # Bottleneck: which transient state has highest expected visits starting from Newbie
    bottleneck_state = None
    bottleneck_value = None
    if "Newbie" in transient_names:
        row = F_df.loc["Newbie", :]
        bottleneck_state = str(row.idxmax())
        bottleneck_value = float(row.max())

    return {
        "transient_idx": transient_idx,
        "absorbing_idx": absorbing_idx,
        "transient_names": transient_names,
        "absorbing_names": absorbing_names,
        "Q_df": Q_df,
        "R_df": R_df,
        "F_df": F_df,
        "B_df": B_df,
        "success": success,
        "tau": tau,
        "bottleneck_state": bottleneck_state,
        "bottleneck_value": bottleneck_value,
    }

baseline_err = None
whatif_err = None

try:
    base = compute_outputs(P)
except Exception as e:
    baseline_err = str(e)
    base = None

P_wi = None
applied = 0.0
if what_if_on and baseline_err is None:
    P_wi, applied = what_if_reduce_newbie_dropout(P, state_names, delta)
    try:
        wi = compute_outputs(P_wi)
    except Exception as e:
        whatif_err = str(e)
        wi = None
else:
    wi = None

# -----------------------------
# Output panels
# -----------------------------
st.divider()
st.subheader("3) State classification")
if baseline_err:
    st.error(f"Could not compute outputs from P. Error: {baseline_err}")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.write("Transient states:")
        st.write(base["transient_names"])
    with c2:
        st.write("Absorbing states:")
        st.write(base["absorbing_names"])

st.divider()
st.subheader("4) Matrices (Q, R, F, B)")

if baseline_err is None:
    a, b = st.columns(2)
    with a:
        st.write("Q (Transient → Transient)")
        st.dataframe(base["Q_df"], use_container_width=True)
        st.write("F = (I − Q)⁻¹ (Fundamental Matrix)")
        st.dataframe(base["F_df"], use_container_width=True)
    with b:
        st.write("R (Transient → Absorbing)")
        st.dataframe(base["R_df"], use_container_width=True)
        st.write("B = F × R (Absorption Probabilities)")
        st.dataframe(base["B_df"], use_container_width=True)

st.divider()
st.subheader("5) Success metric + insights")

if baseline_err is None:
    k1, k2, k3 = st.columns(3)
    k1.metric("P(Newbie → Verified Legend)", f"{base['success']:.4f}" if np.isfinite(base["success"]) else "N/A")
    k2.metric("Life expectancy τ (days/steps)", f"{base['tau']:.2f}" if np.isfinite(base["tau"]) else "N/A")
    if base["bottleneck_state"] is not None:
        k3.metric("Bottleneck (most time spent)", base["bottleneck_state"], f"{base['bottleneck_value']:.2f} expected visits")
    else:
        k3.metric("Bottleneck (most time spent)", "N/A")

st.divider()
st.subheader("6) What-If result (more Legends?)")

if not what_if_on:
    st.info("Enable the What-If toggle to see how reducing Newbie dropout changes the probability of creating Legends.")
elif baseline_err is not None:
    st.warning("Fix baseline matrix issues first.")
elif whatif_err is not None:
    st.error(f"Could not compute What-If outputs. Error: {whatif_err}")
else:
    # "How many more Legends are created?" depends on a cohort size; let user specify.
    cohort = st.number_input("Assumed cohort size (new Newbies)", min_value=1, value=1000, step=100)

    base_p = base["success"]
    wi_p = wi["success"]

    st.write(f"Applied reduction to P(Newbie → Deleted): {applied:.4f}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline P(Newbie → Legend)", f"{base_p:.4f}")
    c2.metric("What-If P(Newbie → Legend)", f"{wi_p:.4f}")
    diff = (wi_p - base_p)
    c3.metric("Absolute increase", f"{diff:.4f}")

    extra_legends = cohort * diff
    st.write(f"Expected extra Verified Legends out of {cohort:,} Newbies: {extra_legends:,.2f}")

    with st.expander("Show What-If P, Q, R, F, B"):
        st.write("What-If P")
        st.dataframe(pd.DataFrame(P_wi, index=state_names, columns=state_names), use_container_width=True)
        st.write("What-If B (Absorption Probabilities)")
        st.dataframe(wi["B_df"], use_container_width=True)

st.caption("Formulas: F = (I − Q)⁻¹ and B = F × R. [file:1]")
