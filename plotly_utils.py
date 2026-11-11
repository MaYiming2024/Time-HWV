"""
plotly_utils.py — shared Plotly volume layout, opacity, and time slider
=============================================================

Shared by plot5 (|ψ|² volume) and plot6 (4πr²|ψ|² volume) so that axes,
backgrounds, fonts, and slider styles stay consistent.
"""

from __future__ import annotations

# Higher-density regions are more opaque; values follow the New ICS convention
OPACITY_SCALE = [
    [0.0, 0.0],
    [0.05, 0.05],
    [0.1, 0.12],
    [0.2, 0.25],
    [0.3, 0.40],
    [0.4, 0.55],
    [0.5, 0.68],
    [0.6, 0.78],
    [0.7, 0.87],
    [0.8, 0.93],
    [0.9, 0.97],
    [1.0, 1.0],
]


MAX_DYNAMIC_POINTS = 800_000


def warn_resolution(resolution, surface_count, dynamic=False):
    """Warn about resolution / isosurface counts without encouraging 50 layers."""
    if resolution < 30:
        print(f"[warn] resolution={resolution} is low; use >= 30.")
    elif resolution < 36:
        print(f"[hint] resolution={resolution} is acceptable; 36-40 is better.")
    if surface_count < 6:
        print(f"[hint] surface_count={surface_count} is small; 8-15 is better.")
    limit = 12 if dynamic else 20
    if surface_count > limit:
        print(f"[hint] surface_count={surface_count} is high; keep <= {limit}.")


def cap_surface_count(surface_count, resolution, dynamic=False, traces=1):
    """Use fast defaults, but respect an explicit user surface_count."""
    if dynamic and traces > 1:
        default = 8
    elif dynamic:
        default = 10
    else:
        default = 16
    if surface_count is None:
        return default
    return int(surface_count)


def plan_dynamic_budget(resolution, frames, traces, high_quality=False):
    """Reduce frames first (scaling dt), then resolution, to fit the budget."""
    resolution = int(resolution)
    frames = int(frames)
    traces = max(1, int(traces))
    notes = []
    if high_quality:
        return resolution, frames, notes

    points = resolution ** 3 * frames * traces
    if points <= MAX_DYNAMIC_POINTS:
        return resolution, frames, notes

    new_frames = max(12, int(frames * MAX_DYNAMIC_POINTS / points))
    if new_frames < frames:
        notes.append(f"frames {frames} -> {new_frames} (dt scaled, same duration)")
        frames = new_frames

    points = resolution ** 3 * frames * traces
    if points > MAX_DYNAMIC_POINTS:
        new_res = max(
            24,
            int((MAX_DYNAMIC_POINTS / max(1, frames * traces)) ** (1.0 / 3.0)),
        )
        if new_res < resolution:
            notes.append(f"resolution {resolution}^3 -> {new_res}^3")
            resolution = new_res

    return resolution, frames, notes


def estimate_html_mb(resolution, frames, traces):
    """Rough HTML size estimate (MB)."""
    points = resolution ** 3 * frames * traces
    return (points * 5.4 + resolution ** 3 * 32) / 1e6 + 4.0


def apply_scene(fig, background="black", title="", show_labels=True,
                lim=None, font_size=12):
    """Apply a consistent 3D scene, background, font, and title."""
    bg = "black" if str(background).lower() == "black" else "white"
    fg = "white" if bg == "black" else "black"

    axis = dict(color=fg, gridcolor="rgba(128,128,128,0.35)")
    if lim is not None:
        axis["range"] = [-lim, lim]

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X (a0)", **axis),
            yaxis=dict(title="Y (a0)", **axis),
            zaxis=dict(title="Z (a0)", **axis),
            bgcolor=bg,
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        title=title if show_labels else "",
        font=dict(color=fg, size=font_size),
        margin=dict(l=0, r=0, t=50 if show_labels else 10, b=0),
    )
    return fig


def add_time_controls(fig, n_frames, dt, background="black", interval_ms=200):
    """Add a Plotly time slider and play/pause buttons."""
    bg = "black" if str(background).lower() == "black" else "white"
    fg = "white" if bg == "black" else "black"

    slider_steps = []
    for k in range(n_frames):
        slider_steps.append(
            dict(
                method="animate",
                args=[
                    [f"frame{k}"],
                    dict(mode="immediate",
                         frame=dict(duration=0, redraw=True)),
                ],
                label=f"{k * dt:.1f}",
            )
        )

    sliders = [
        dict(
            steps=slider_steps,
            transition=dict(duration=0),
            x=0, y=0, len=0.9,
            currentvalue=dict(
                font=dict(size=14, color=fg),
                prefix="Time: ", suffix=" a.u.", visible=True,
            ),
            font=dict(color=fg),
            bgcolor="rgba(0,0,0,0.6)" if bg == "black" else "rgba(255,255,255,0.6)",
        )
    ]

    # Plotly's default button face is light, so use dark text on a light
    # button (inverse contrast) on both dark and light scenes.
    button_bg = "rgba(255,255,255,0.96)"
    button_fg = "black"

    updatemenus = [
        dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play", method="animate",
                    args=[None, dict(frame=dict(duration=interval_ms,
                                                redraw=True),
                                     fromcurrent=True, mode="immediate")],
                ),
                dict(
                    label="Pause", method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode="immediate",
                                       transition=dict(duration=0))],
                ),
            ],
            font=dict(color=button_fg, size=16),
            bgcolor=button_bg,
            bordercolor="black",
            borderwidth=1,
            pad=dict(t=6, b=6, l=10, r=10),
            x=0.0, y=1.04,
        )
    ]

    fig.update_layout(sliders=sliders, updatemenus=updatemenus)
    return fig
