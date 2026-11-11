"""
plot3.py — 3D surface plot (height field) of probability density |ψ|² (static + time-dependent)
====================================================

Based on the New ICS "3 3D surface |ψ|²" view.

- static: a height field of |ψ|² on a selected cross-section, normalized to its maximum;
- dynamic: a 1×3 panel of Re ψ / Im ψ / |ψ|² with a fixed z-range, showing phase evolution.

Note: a 3D surface plot is a height field on a 2D cross-section, not a volume rendering; use plot5/plot6 for a true 3D distribution.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from core import build_plane_data, plane_meta
from palettes import cmap_for, is_magnitude, resolve_scheme
from plot_common import (
    add_nodal_note_3d,
    add_nodal_note_fig,
    normalize_nonneg,
    set_3d_view,
    style_3d_axis,
    style_colorbar,
    surface_stride,
)


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------
def plot3_surface_density(
    calc,
    n,
    l,
    m,
    plane="xz",
    scale=None,
    grid_size=300,
    a0=None,
    scheme="classic",
    show_guides=True,
    show_labels=True,
    orbital_type="complex",
    real_component="cos",
    rel_eps=1e-6,
    vmax=None,
    background="white",
    view=(28, -60),
):
    """Static 3D surface plot (height field) of |ψ|² on a selected cross-section. Returns a matplotlib Figure."""
    data = build_plane_data(
        calc, n, l, m, plane=plane, scale=scale, grid_size=grid_size,
        orbital_type=orbital_type, real_component=real_component,
    )
    d, _ = normalize_nonneg(data["density"], rel_eps=rel_eps, vmax=vmax)
    meta = plane_meta(plane)

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor(background)
    ax = fig.add_subplot(111, projection="3d")

    stride = surface_stride(grid_size, res_factor=1.0, max_points=110)
    C1 = data["C1"][::stride, ::stride]
    C2 = data["C2"][::stride, ::stride]
    d_plot = d[::stride, ::stride]

    surf = ax.plot_surface(
        C1, C2, d_plot, cmap=cmap_for(scheme, "density"),
        vmin=0.0, vmax=1.0, alpha=0.85, linewidth=0, antialiased=True,
    )
    ax.set_zlim(0.0, 1.08)
    style_3d_axis(
        ax, background=background, show_labels=show_labels,
        scale=data["scale"], xlabel=data["xlabel"], ylabel=data["ylabel"],
        zlabel=r"$|\psi|^2$",
    )
    set_3d_view(ax, *view)
    if show_labels:
        ax.set_title(
            rf"3D Surface (Height Field) of $|\psi|^2$ — n={n}, l={l}, m={m} "
            rf"({meta['name']} cross-section)",
            fontsize=12,
        )
    if data["nodal_plane"]:
        add_nodal_note_3d(ax, background=background)

    if not show_guides:
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.08)
    if show_labels:
        cbar.set_label(r"$|\psi|^2$ (relative, max = 1)")
    style_colorbar(cbar, background)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dynamic
# ---------------------------------------------------------------------------
def plot3_surface_density_dynamic(
    calc,
    n,
    l,
    m,
    plane="xz",
    scale=None,
    grid_size=300,
    a0=None,
    scheme="classic",
    dt=1.0,
    frames=100,
    interval=50,
    show_labels=True,
    background="black",
    res_factor=0.2,
    orbital_type="complex",
    real_component="cos",
):
    """
    Time-dependent 3D surface plot animation: Re ψ, Im ψ, and |ψ|².

    Returns (animation, figure).
    """
    data = build_plane_data(
        calc, n, l, m, plane=plane, scale=scale, grid_size=grid_size,
        orbital_type=orbital_type, real_component=real_component,
    )
    psi = data["psi"]
    prob = data["density"]
    mag = is_magnitude(scheme)

    psi_real = np.real(psi)
    psi_imag = np.imag(psi)
    if mag:
        series = [np.abs(psi_real), np.abs(psi_imag), prob]
        lims = [(0.0, 1.15)] * 3
        titles = ["|Re ψ|", "|Im ψ|", r"$|\psi|^2$"]
        cmaps = [cmap_for(scheme, "real"), cmap_for(scheme, "imag"),
                 cmap_for(scheme, "density")]
    else:
        series = [psi_real, psi_imag, prob]
        lims = [(-1.15, 1.15), (-1.15, 1.15), (0.0, 1.15)]
        titles = ["Re ψ", "Im ψ", r"$|\psi|^2$"]
        cmaps = [cmap_for(scheme, "real"), cmap_for(scheme, "imag"),
                 cmap_for(scheme, "density")]

    stride = surface_stride(grid_size, res_factor=res_factor, max_points=110)
    C1 = data["C1"][::stride, ::stride]
    C2 = data["C2"][::stride, ::stride]
    series_plot = [arr[::stride, ::stride] for arr in series]

    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor(background)
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    fg = "white" if str(background).lower() == "black" else "black"
    meta = plane_meta(plane)

    def draw_panel(ax, arr, title, cmap, zlim):
        ax.cla()
        ax.plot_surface(
            C1, C2, arr, cmap=cmap, vmin=zlim[0], vmax=zlim[1],
            alpha=0.85, linewidth=0, antialiased=True,
        )
        ax.set_zlim(*zlim)
        style_3d_axis(
            ax, background=background, show_labels=show_labels,
            scale=data["scale"], xlabel=data["xlabel"],
            ylabel=data["ylabel"], zlabel="amplitude / density",
        )
        set_3d_view(ax, 28, -60)
        ax.set_title(title, color=fg, fontsize=11)

    for ax, arr, title, cmap, zlim in zip(
        axes, series_plot, titles, cmaps, lims,
    ):
        draw_panel(ax, arr, title, cmap, zlim)

    if show_labels:
        fig.suptitle(
            rf"n={n}, l={l}, m={m}  ({meta['name']} cross-section) — "
            rf"3D surface (height field) phase evolution",
            color=fg, fontsize=15,
        )
    time_text = fig.text(
        0.02, 0.95, "", color="cyan" if fg == "white" else "darkblue",
        fontsize=13,
    )
    if data["nodal_plane"]:
        add_nodal_note_fig(fig, background=background)

    e_n = -0.5 / (n * n)

    def update(frame):
        t = frame * dt
        psi_t = psi * np.exp(-1j * e_n * t)
        re_t = np.real(psi_t)[::stride, ::stride]
        im_t = np.imag(psi_t)[::stride, ::stride]
        prob_t = (np.abs(psi_t) ** 2)[::stride, ::stride]
        if mag:
            arrays = [np.abs(re_t), np.abs(im_t), prob_t]
        else:
            arrays = [re_t, im_t, prob_t]
        for ax, arr, title, cmap, zlim in zip(
            axes, arrays, titles, cmaps, lims,
        ):
            draw_panel(ax, arr, title, cmap, zlim)
        time_text.set_text(f"t = {t:.1f} a.u.")
        return [time_text]

    from matplotlib.animation import FuncAnimation

    ani = FuncAnimation(
        fig, update, frames=int(max(1, frames)), interval=interval, blit=False,
    )
    fig.tight_layout()
    return ani, fig
