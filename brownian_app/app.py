"""
app.py
------
Streamlit front-end for the Brownian Motion / Diffusion Explorer.
Run with: streamlit run app.py
"""

import pandas as pd
import streamlit as st
import numpy as np

from simulation import generate_walkers
from analysis import fit_diffusion, final_displacement_stats
from viz import (
    fig_trajectories_3d,
    fig_msd,
    fig_displacement_hist,
    fig_displacement_time,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Diffusion Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0f0f13; }
    section[data-testid="stSidebar"] { background-color: #16161d; }

    [data-testid="metric-container"] {
        background: #1e1e2a;
        border: 1px solid #2a2a3a;
        border-radius: 10px;
        padding: 12px 16px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #16161d;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #888;
        font-size: 13px;
        padding: 6px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2a2a3a !important;
        color: #fff !important;
    }

    h1 { color: #e8e8f0 !important; font-size: 1.6rem !important; }
    h2 { color: #b0b0c4 !important; font-size: 1.1rem !important; }
    h3 { color: #9090a8 !important; font-size: 0.95rem !important; }

    .info-box {
        background: #1a1a28;
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 13px;
        color: #aaa;
        line-height: 1.6;
        margin-bottom: 16px;
    }
    .info-box b { color: #c7c7e0; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️  Simulation")

    n_walkers  = st.slider("Number of particles", 1, 20, 5)
    n_steps    = st.slider("Steps per particle", 100, 5000, 600, step=100)
    step_size  = st.slider("Step size σ", 0.5, 3.0, 1.0, step=0.1)
    seed_val   = st.number_input("Random seed", min_value=0, max_value=9999,
                                  value=42, step=1)

    st.divider()
    st.markdown("## 🎬  Visualisation")
    trail_pct      = st.slider("Trail resolution (%)", 10, 100, 80, step=10,
                                help="Reduce for faster rendering with many steps")
    show_endpoints = st.checkbox("Show start / end markers", value=True)

    st.divider()
    st.markdown("## 🌍  Real-world context")
    context = st.selectbox("Frame as:", [
        "Nanoparticle in solution",
        "Drug delivery through tissue",
        "Pollen grain in water (Einstein 1905)",
        "Polymer chain dynamics",
        "Financial asset random walk",
    ])

    st.divider()
    run = st.button("▶  Run simulation", use_container_width=True, type="primary")


# ── Context blurbs ─────────────────────────────────────────────────────────────

CONTEXT_TEXT = {
    "Nanoparticle in solution": (
        "Each walker models a <b>nanoparticle</b> (e.g. silica, gold) suspended in a solvent. "
        "The diffusion coefficient D relates to particle radius r and viscosity η via the "
        "Stokes–Einstein equation: <b>D = k_B T / 6πηr</b>."
    ),
    "Drug delivery through tissue": (
        "In drug delivery, molecules diffuse through tissue extracellular matrix. "
        "The MSD profile reveals how quickly a drug reaches its target — "
        "anomalous diffusion (MSD ∝ t^α, α≠1) indicates confinement or obstacles."
    ),
    "Pollen grain in water (Einstein 1905)": (
        "This is the original <b>Brownian motion</b> observed by Robert Brown in 1827 and "
        "explained by Einstein in 1905. His formula linked diffusivity to molecular-scale "
        "thermal energy, providing indirect proof of atoms."
    ),
    "Polymer chain dynamics": (
        "Each step models a monomer in a <b>freely-jointed polymer chain</b>. "
        "The end-to-end distance scales as √N (Flory's ideal chain), "
        "and D captures the chain's centre-of-mass mobility."
    ),
    "Financial asset random walk": (
        "Under the efficient-market hypothesis, log-price increments are i.i.d. — "
        "a discrete random walk. The MSD slope encodes <b>volatility²</b>, and "
        "deviations from linearity signal auto-correlation or momentum effects."
    ),
}


# ── Session-state cache ────────────────────────────────────────────────────────

# Re-run the simulation only when the button is pressed, not on every widget
# interaction (Streamlit reruns the whole script on any UI event).
if "sim" not in st.session_state or run:
    with st.spinner("Running simulation…"):
        st.session_state.sim = generate_walkers(
            n_walkers=n_walkers,
            n_steps=n_steps,
            step_size=step_size,
            seed=int(seed_val),
        )

sim = st.session_state.sim


# ── Derived quantities ─────────────────────────────────────────────────────────

fit      = fit_diffusion(sim["t"], sim["msd"])
final_st = final_displacement_stats(sim["displacement"])


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("# 🔬 Diffusion Explorer")
st.markdown(
    f'<div class="info-box">{CONTEXT_TEXT[context]}</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Particles",           sim["n_walkers"])
c2.metric("Steps / particle",    sim["n_steps"])
c3.metric("Diffusion coeff. D",  f"{fit['D']:.3f}", help="Fitted from MSD = 6Dt")
c4.metric("Fit R²",              f"{fit['r_squared']:.4f}")

st.divider()


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_3d, tab_physics, tab_about = st.tabs([
    "  🧊  3D Trajectories  ",
    "  📈  Physics Analysis  ",
    "  ℹ️  About  ",
])


# ── Tab 1: 3D Trajectories ────────────────────────────────────────────────────

with tab_3d:
    st.markdown("#### Particle trajectories in 3D space")
    st.markdown(
        "<p style='font-size:13px;color:#777;margin-top:-8px'>"
        "Trail colour encodes cumulative displacement from origin (Plasma scale). "
        "Click + drag to rotate · scroll to zoom · double-click to reset."
        "</p>",
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        fig_trajectories_3d(
            positions=sim["positions"],
            displacement=sim["displacement"],
            trail_frac=trail_pct / 100,
            show_start_end=show_endpoints,
        ),
        use_container_width=True,
    )

    with st.expander("📋  Per-walker summary"):
        rows = [
            {
                "Walker":    w + 1,
                "Final x":   round(float(sim["positions"][w, -1, 0]), 3),
                "Final y":   round(float(sim["positions"][w, -1, 1]), 3),
                "Final z":   round(float(sim["positions"][w, -1, 2]), 3),
                "|r| final": round(float(sim["displacement"][w, -1]), 3),
            }
            for w in range(sim["n_walkers"])
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Tab 2: Physics ────────────────────────────────────────────────────────────

with tab_physics:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Mean squared displacement")
        st.markdown(
            "<p style='font-size:13px;color:#777;margin-top:-8px'>"
            "⟨r²⟩ = 6Dt in 3D. Dashed line shows the linear fit."
            "</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_msd(sim["t"], sim["msd"], fit),
                        use_container_width=True)

    with col_r:
        st.markdown("#### Displacement over time (per walker)")
        st.markdown(
            "<p style='font-size:13px;color:#777;margin-top:-8px'>"
            "Individual |r(t)| traces with the theoretical √t envelope."
            "</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_displacement_time(sim["t"], sim["displacement"]),
                        use_container_width=True)

    st.markdown("#### Final displacement distribution")
    st.markdown(
        "<p style='font-size:13px;color:#777;margin-top:-8px'>"
        f"Histogram of |r| at t = {sim['n_steps']}. "
        f"Expected RMS displacement √N = {final_st['theoretical_rms']:.1f}."
        "</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        fig_displacement_hist(final_st["values"], final_st["theoretical_rms"]),
        use_container_width=True,
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("D (fitted)",           f"{fit['D']:.4f}")
    col_b.metric("Mean final |r|",       f"{final_st['mean']:.2f}")
    col_c.metric("Theoretical RMS √N",   f"{final_st['theoretical_rms']:.2f}")


# ── Tab 3: About ──────────────────────────────────────────────────────────────

with tab_about:
    st.markdown("""
### About this app

This app simulates **3D Brownian motion** (isotropic random walks on the unit sphere)
and measures the emergent diffusion coefficient from the ensemble MSD.

#### Physics
- Each step is drawn uniformly on the unit sphere using the correct parameterisation:
  cos φ ~ Uniform(−1, 1) to avoid pole-oversampling.
- In 3D, the mean squared displacement grows as **⟨r²⟩ = 6Dt**, where D is the
  diffusion coefficient. The fit uses the linear middle segment of the MSD curve
  to avoid edge artefacts.

#### Stack
- **Simulation**: NumPy (vectorised over all walkers simultaneously)
- **Analysis**: SciPy linear regression for D fitting
- **Visualisation**: Plotly (interactive 3D + 2D charts)
- **App**: Streamlit

#### Real-world relevance
Brownian motion underlies diffusion in soft matter physics, drug delivery modelling,
polymer science, colloidal systems, and financial time-series analysis.

---
*Built as part of a data science portfolio — Francesco Conte, 2025.*
""")
