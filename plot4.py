"""
plot4.py — 3D surface plot (height field) of shell density 4πr²|ψ|² (static + time-dependent)
=========================================================

Based on the New ICS "4 3D shell surface" view.

- static: optional RectBivariateSpline interpolation to a finer grid;
- dynamic: a 1×3 panel of Re ψ / Im ψ / 4πr²|ψ|² with a fixed z-range.

Shell density raises the weight of outer radial shells and is the preferred 3D surface-plot
representation for multi-node states with large n and small l.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import RectBivariateSpline
    INTERP_AVAILABLE = True
except ImportError:  # pragma: no cover
    INTERP_AVAILABLE = False

from core import build_plane_data, plane_meta, shell_from_density
from palettes import cmap_for, is_magnitude
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
def plot4_surface_shell(
    calc,
    n,
    l,
    m,
    plane="xz",
    scale=None,
    grid_size=400,
    a0=None,
    scheme="classic",
    show_box=True,
    show_labels=True,
    orbital_type="complex",
    real_component="cos",
    rel_eps=1e-6,
    vmax=None,
    use_interp=True,
    background="white",
    view=(28, -60),
):
    """
    Static 3D shell-density surface (height field) on a selected cross-section.

    Returns a matplotlib Figure.
    """
    a0 = a0 or calc.a0
    data = build_plane_data(
        calc, n, l, m, plane=plane, scale=scale, grid_size=grid_size,
        orbital_type=orbital_type, real_component=real_component,
    )
    shell = shell_from_density(data["r"], data["density"], a0=a0)
    d, _ = normalize_nonneg(shell, rel_eps=rel_eps, vmax=vmax)
    meta = plane_meta(plane)

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor(background)
    ax = fig.add_subplot(111, projection="3d")

    C1, C2 = data["C1"], data["C2"]
    if use_interp and INTERP_AVAILABLE:
        try:
            x = C1[0, :]
            z = C2[:, 0]
            interp = RectBivariateSpline(z, x, d)
            n_new = 1200
            x_new = np.linspace(x.min(), x.max(), n_new)
            z_new = np.linspace(z.min(), z.max(), n_new)
            C1_new, C2_new = np.meshgrid(x_new, z_new)
            shell_new = interp(z_new, x_new)
            shell_new = np.clip(shell_new, 0.0, 1.05)
            C1_plot, C2_plot, d_plot = C1_new, C2_new, shell_new
        except Exception:
            stride = surface_stride(grid_size, 1.0)
            C1_plot = C1[::stride, ::stride]
            C2_plot = C2[::stride, ::stride]
            d_plot = d[::stride, ::stride]
    else:
        stride = surface_stride(grid_size, 1.0, max_points=200)
        C1_plot = C1[::stride, ::stride]
        C2_plot = C2[::stride, ::stride]
        d_plot = d[::stride, ::stride]

    surf = ax.plot_surface(
        C1_plot, C2_plot, d_plot, cmap=cmap_for(scheme, "shell"),
        vmin=0.0, vmax=1.0, alpha=0.85, linewidth=0, antialiased=True,
    )
    ax.set_zlim(0.0, 1.08)
    style_3d_axis(
        ax, background=background, show_labels=show_labels,
        scale=data["scale"], xlabel=data["xlabel"], ylabel=data["ylabel"],
        zlabel="shell density",
    )
    set_3d_view(ax, *view)
    if show_labels:
        ax.set_title(
            rf"3D Shell Density (Height Field) $4\pi r^2|\psi|^2$ — "
            rf"n={n}, l={l}, m={m} ({meta['name']} cross-section)",
            fontsize=12,
        )
    if data["nodal_plane"]:
        add_nodal_note_3d(ax, background=background)

    if not show_box:
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.08)
    if show_labels:
        cbar.set_label(r"$4\pi r^2|\psi|^2$ (relative, max = 1)")
    style_colorbar(cbar, background)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dynamic
# ---------------------------------------------------------------------------
def plot4_surface_shell_dynamic(
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
    Time-dependent 3D shell-density surface plot animation: Re ψ, Im ψ, and 4πr²|ψ|².

    Returns (animation, figure).
    """
    a0 = a0 or calc.a0
    data = build_plane_data(
        calc, n, l, m, plane=plane, scale=scale, grid_size=grid_size,
        orbital_type=orbital_type, real_component=real_component,
    )
    psi = data["psi"]
    shell = shell_from_density(data["r"], data["density"], a0=a0)
    mag = is_magnitude(scheme)

    psi_real = np.real(psi)
    psi_imag = np.imag(psi)
    if mag:
        series = [np.abs(psi_real), np.abs(psi_imag), shell]
        titles = ["|Re ψ|", "|Im ψ|", r"$4\pi r^2|\psi|^2$"]
        lims = [(0.0, 1.15)] * 3
    else:
        series = [psi_real, psi_imag, shell]
        titles = ["Re ψ", "Im ψ", r"$4\pi r^2|\psi|^2$"]
        lims = [(-1.15, 1.15), (-1.15, 1.15), (0.0, 1.15)]
    cmaps = [
        cmap_for(scheme, "real"), cmap_for(scheme, "imag"),
        cmap_for(scheme, "shell"),
    ]

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
            ylabel=data["ylabel"], zlabel="amplitude / shell density",
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
            rf"3D shell-density surface (height field) phase evolution",
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
        shell_t = shell[::stride, ::stride]
        if mag:
            arrays = [np.abs(re_t), np.abs(im_t), shell_t]
        else:
            arrays = [re_t, im_t, shell_t]
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
