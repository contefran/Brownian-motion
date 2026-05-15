"""
viz.py
------
All Plotly figures. Returns go.Figure objects; no Streamlit calls here.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

WALKER_COLORS = px.colors.qualitative.Bold  # cycles for N > 10


# ── 3D Trajectory ──────────────────────────────────────────────────────────────

def fig_trajectories_3d(
    positions: np.ndarray,
    displacement: np.ndarray,
    trail_frac: float = 1.0,
    show_start_end: bool = True,
) -> go.Figure:
    """
    3D line plot of all walker trajectories.

    trail_frac: float in (0, 1] — downsample steps for faster rendering.
    Trail colour encodes cumulative displacement (Plasma colorscale).
    """
    n_walkers, n_steps_1, _ = positions.shape
    step_show = max(1, int(n_steps_1 * trail_frac))
    idx = np.round(np.linspace(0, n_steps_1 - 1, step_show)).astype(int)

    fig = go.Figure()

    for w in range(n_walkers):
        pos   = positions[w, idx]
        dist  = displacement[w, idx]
        color = WALKER_COLORS[w % len(WALKER_COLORS)]

        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode="lines",
            line=dict(width=2, color=dist, colorscale="Plasma",
                      cmin=0, cmax=dist.max()),
            name=f"Walker {w+1}",
            showlegend=True,
            hovertemplate=(
                f"<b>Walker {w+1}</b><br>"
                "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>"
            ),
        ))

        if show_start_end:
            fig.add_trace(go.Scatter3d(
                x=[pos[0, 0]], y=[pos[0, 1]], z=[pos[0, 2]],
                mode="markers",
                marker=dict(size=5, color="white", symbol="circle",
                            line=dict(color=color, width=2)),
                name=f"W{w+1} start",
                showlegend=False,
            ))
            fig.add_trace(go.Scatter3d(
                x=[pos[-1, 0]], y=[pos[-1, 1]], z=[pos[-1, 2]],
                mode="markers",
                marker=dict(size=7, color=color, symbol="diamond"),
                name=f"W{w+1} end",
                showlegend=False,
            ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=600,
        scene=dict(
            xaxis=dict(title="X", gridcolor="#333", showbackground=False),
            yaxis=dict(title="Y", gridcolor="#333", showbackground=False),
            zaxis=dict(title="Z", gridcolor="#333", showbackground=False),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=11)),
    )
    return fig


# ── MSD plot ───────────────────────────────────────────────────────────────────

def fig_msd(t: np.ndarray, msd: np.ndarray, fit: dict) -> go.Figure:
    """MSD vs time with theoretical 6Dt overlay."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, y=msd,
        mode="lines",
        name="Measured MSD",
        line=dict(color="#818CF8", width=2),
        hovertemplate="t=%{x}<br>MSD=%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=fit["fit_msd"],
        mode="lines",
        name=f"Fit: 6Dt  (D = {fit['D']:.3f})",
        line=dict(color="#F472B6", width=2, dash="dash"),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(l=50, r=20, t=20, b=50),
        xaxis=dict(title="Step (t)", gridcolor="#2a2a2a"),
        yaxis=dict(title="⟨r²⟩  (MSD)", gridcolor="#2a2a2a"),
        legend=dict(font=dict(size=11)),
    )
    return fig


# ── Displacement histogram ─────────────────────────────────────────────────────

def fig_displacement_hist(final_vals: np.ndarray, theoretical_rms: float) -> go.Figure:
    """Histogram of final walker displacements + expected √N reference line."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=final_vals,
        nbinsx=max(10, len(final_vals) // 3),
        name="Final |r|",
        marker_color="#34D399",
        opacity=0.75,
    ))
    fig.add_vline(
        x=theoretical_rms,
        line_dash="dash", line_color="#FBBF24",
        annotation_text=f"√N = {theoretical_rms:.1f}",
        annotation_position="top right",
        annotation_font=dict(color="#FBBF24", size=11),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=50, r=20, t=20, b=50),
        xaxis=dict(title="Final displacement |r|", gridcolor="#2a2a2a"),
        yaxis=dict(title="Count", gridcolor="#2a2a2a"),
        showlegend=False,
    )
    return fig


# ── Displacement over time (per walker) ───────────────────────────────────────

def fig_displacement_time(t: np.ndarray, displacement: np.ndarray) -> go.Figure:
    """|r(t)| vs t for each walker, + √t theoretical envelope."""
    fig = go.Figure()

    n_walkers = displacement.shape[0]
    for w in range(n_walkers):
        color = WALKER_COLORS[w % len(WALKER_COLORS)]
        fig.add_trace(go.Scatter(
            x=t, y=displacement[w],
            mode="lines",
            name=f"Walker {w+1}",
            line=dict(color=color, width=1.5),
            opacity=0.8,
        ))

    fig.add_trace(go.Scatter(
        x=t, y=np.sqrt(t.astype(float)),
        mode="lines",
        name="√t  (theory)",
        line=dict(color="white", width=2, dash="dot"),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(l=50, r=20, t=20, b=50),
        xaxis=dict(title="Step (t)", gridcolor="#2a2a2a"),
        yaxis=dict(title="|r(t)|  displacement", gridcolor="#2a2a2a"),
        legend=dict(font=dict(size=11)),
    )
    return fig
