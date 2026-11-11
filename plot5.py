"""
plot5.py — 3D volume rendering of probability density |ψ|² (Plotly static + time slider)
==============================================================

Based on the New ICS "5 volume rendering |ψ|²" view.

- static: Plotly go.Volume with Gaussian smoothing and an opacity map;
- dynamic: time slider + play/pause, with real / imag / both modes.

Known limitation: Plotly volume rendering uses discrete voxels and screen-space ray marching,
so oblique views can show voxel staircase artifacts. Typical values are
smooth_sigma = 1.0 and opacity = 0.1.

If the outer shells of a multi-node radial state are suppressed after normalization, use plot6
(volume rendering of the shell density 4πr²|ψ|²).
"""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

from core import build_volume_data
from palettes import cmap_for, is_magnitude, plotly_scale
from plotly_utils import (
    OPACITY_SCALE,
    add_time_controls,
    apply_scene,
    cap_surface_count,
    estimate_html_mb,
    plan_dynamic_budget,
    warn_resolution,
)


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------
def plot5_volume_density(
    calc,
    n,
    l,
    m,
    scale,
    a0=None,
    resolution=50,
    scheme="classic",
    background="black",
    show_labels=True,
    zoom_factor=0.8,
    smooth_sigma=1.0,
    surface_count=None,
    opacity=0.1,
    orbital_type="complex",
    real_component="cos",
):
    """Static |ψ|² volume rendering. Returns a Plotly Figure (caller uses .show() / .write_html())."""
    surface_count = cap_surface_count(surface_count, resolution, dynamic=False)
    warn_resolution(resolution, surface_count, dynamic=False)
    print(f"Building |psi|^2 volume ({resolution}^3 grid, {surface_count} isosurfaces)...")
    start = time.time()

    X, Y, Z, r, mask, psi, density, shell = build_volume_data(
        calc, n, l, m, scale, resolution=resolution,
        orbital_type=orbital_type, real_component=real_component,
    )
    Xa, Ya, Za = X / calc.a0, Y / calc.a0, Z / calc.a0
    print(f"Gaussian smoothing sigma={smooth_sigma} ...")
    density_smooth = gaussian_filter(
        density, sigma=smooth_sigma, mode="constant", cval=0.0,
    )
    dmax = float(np.max(density_smooth))
    if dmax > 0:
        density_smooth = density_smooth / dmax

    fig = go.Figure(
        data=go.Volume(
            x=Xa.flatten(), y=Ya.flatten(), z=Za.flatten(),
            value=density_smooth.flatten(),
            isomin=0.0, isomax=1.0,
            opacity=float(opacity),
            opacityscale=OPACITY_SCALE,
            colorscale=plotly_scale(cmap_for(scheme, "density")),
            surface_count=surface_count,
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=True,
            colorbar=dict(title=r"$|\psi|^2$ (norm)", x=0.95),
        )
    )

    lim = scale / calc.a0 * zoom_factor
    title = (
        rf"Volume Rendering of $|\psi|^2$  "
        rf"(n={n}, l={l}, m={m}; resolution={resolution}, "
        rf"surface_count={surface_count}, σ={smooth_sigma})"
    )
    apply_scene(
        fig, background=background, title=title,
        show_labels=show_labels, lim=lim,
    )

    if show_labels and resolution < 40:
        fig.add_annotation(
            text=f"Low resolution (res={resolution}); voxel artifacts may be visible",
            xref="paper", yref="paper", x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=11,
                      color="white" if str(background).lower() == "black" else "black"),
            bgcolor="rgba(255,80,80,0.6)",
            bordercolor="red", borderwidth=1, align="left",
        )

    print(f"Volume rendering finished in {time.time() - start:.2f} s.")
    return fig


# ---------------------------------------------------------------------------
# Dynamic
# ---------------------------------------------------------------------------
def plot5_volume_density_dynamic(
    calc,
    n,
    l,
    m,
    scale,
    a0=None,
    resolution=40,
    dt=0.5,
    frames=50,
    mode="real",
    scheme="classic",
    background="black",
    show_labels=True,
    zoom_factor=0.8,
    smooth_sigma=1.0,
    surface_count=None,
    opacity=0.1,
    orbital_type="complex",
    real_component="cos",
):
    """
    Time-dependent |ψ|² volume rendering (Plotly slider).

    mode: 'real' | 'imag' | 'both'
        real / imag shows one component; both overlays the real and imaginary parts.
    Returns a Plotly Figure.
    """
    mode = str(mode).lower()
    if mode not in ("real", "imag", "both"):
        mode = "real"
    frames = max(1, int(frames))
    requested_frames = frames
    traces = 2 if mode == "both" else 1
    resolution, frames, notes = plan_dynamic_budget(resolution, frames, traces)
    if frames != requested_frames:
        dt = dt * requested_frames / frames
    surface_count = cap_surface_count(surface_count, resolution, dynamic=True,
                                      traces=traces)
    warn_resolution(resolution, surface_count, dynamic=True)
    for note in notes:
        print(f"[perf] {note}")
    print(f"[hint] estimated HTML ~{estimate_html_mb(resolution, frames, traces):.0f} MB")

    print(
        f"Building time-dependent volume ({resolution}^3 grid, {frames} frames, mode={mode})..."
    )
    start = time.time()

    X, Y, Z, r, mask, psi, density, shell = build_volume_data(
        calc, n, l, m, scale, resolution=resolution,
        orbital_type=orbital_type, real_component=real_component,
    )
    Xa, Ya, Za = X / calc.a0, Y / calc.a0, Z / calc.a0
    mag = is_magnitude(scheme)
    e_n = -0.5 / (n * n)

    real_frames = []
    imag_frames = []
    max_real = 0.0
    max_imag = 0.0

    for frame in range(int(frames)):
        t = frame * dt
        psi_t = psi * np.exp(-1j * e_n * t)
        re_t = np.real(psi_t)
        im_t = np.imag(psi_t)
        if mag:
            re_t = np.abs(re_t)
            im_t = np.abs(im_t)
        re_s = gaussian_filter(re_t, sigma=smooth_sigma,
                               mode="constant", cval=0.0)
        im_s = gaussian_filter(im_t, sigma=smooth_sigma,
                               mode="constant", cval=0.0)
        real_frames.append(re_s.astype(np.float32).flatten())
        imag_frames.append(im_s.astype(np.float32).flatten())
        max_real = max(max_real, float(np.max(np.abs(re_s))))
        max_imag = max(max_imag, float(np.max(np.abs(im_s))))

    if mode == "real":
        gmax = max_real or 1.0
    elif mode == "imag":
        gmax = max_imag or 1.0
    else:
        gmax = max(max_real, max_imag) or 1.0

    if mode in ("imag", "both") and max_imag < 1e-12:
        print("[hint] Im(psi) is approximately 0 for this orbital/component (real orbital or m=0);"
              "the imag panel will be nearly empty.")

    if mag:
        isomin, isomax = 0.0, gmax
    else:
        isomin, isomax = -gmax, gmax

    fig = go.Figure()
    colorscale_real = plotly_scale(cmap_for(scheme, "real"))
    colorscale_imag = plotly_scale(cmap_for(scheme, "imag"))

    def volume_trace(values, colorscale, colorbar_title, colorbar_x):
        return go.Volume(
            x=Xa.flatten(), y=Ya.flatten(), z=Za.flatten(),
            value=values,
            isomin=isomin, isomax=isomax,
            opacity=float(opacity),
            opacityscale=OPACITY_SCALE,
            colorscale=colorscale,
            surface_count=surface_count,
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=True,
            colorbar=dict(title=colorbar_title, x=colorbar_x, y=0.5),
        )

    if mode == "real":
        fig.add_trace(volume_trace(real_frames[0], colorscale_real, "Re(ψ)", 0.95))
        trace_plan = [("real", 0)]
    elif mode == "imag":
        fig.add_trace(volume_trace(imag_frames[0], colorscale_imag, "Im(ψ)", 0.95))
        trace_plan = [("imag", 0)]
    else:
        fig.add_trace(volume_trace(real_frames[0], colorscale_real, "Re(ψ)", 0.85))
        fig.add_trace(volume_trace(imag_frames[0], colorscale_imag, "Im(ψ)", 0.15))
        trace_plan = [("real", 0), ("imag", 1)]

    plot_frames = []
    for i in range(int(frames)):
        frame_data = []
        for which, _ in trace_plan:
            values = real_frames[i] if which == "real" else imag_frames[i]
            frame_data.append(go.Volume(value=values))
        plot_frames.append(go.Frame(data=frame_data, name=f"frame{i}"))
    fig.frames = plot_frames

    add_time_controls(fig, int(frames), dt, background=background,
                      interval_ms=200)

    lim = scale / calc.a0 * zoom_factor
    label = {"real": "Real Part", "imag": "Imaginary Part",
             "both": "Real + Imag"}[mode]
    title = (
        rf"Dynamic volume rendering ({label}) — n={n}, l={l}, m={m}"
    )
    apply_scene(
        fig, background=background, title=title,
        show_labels=show_labels, lim=lim,
    )
    print(f"Time-dependent volume finished in {time.time() - start:.2f} s.")
    return fig
