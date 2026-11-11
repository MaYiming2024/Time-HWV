"""
plot1.py — 2D cross-section probability density |ψ|² (static + time-dependent)
====================================================

Based on the New ICS "1 enhanced 2D" view, with:
- selectable xy / xz / yz cross-sections;
- selectable color schemes, including color-vision-friendly options;
- dynamic version: a 1×3 panel showing the phase evolution of Re ψ, Im ψ, and |ψ|².

Physics note
--------
For a stationary state, ψ(t) = ψ(0) exp(-i E_n t), so |ψ|² is time-independent;
the constant |ψ|² panel in the animation is correct; only the Re/Im panels rotate with phase.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from core import build_plane_data, plane_meta
from palettes import cmap_for, resolve_scheme
from plot_common import (
    add_info_box,
    add_nodal_note_2d,
    draw_classical_circle,
    draw_contours,
    normalize_nonneg,
    phase_animation,
    phase_animation_three_planes,
    style_2d_axis,
    style_colorbar,
)


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------
def plot1_density_2d(
    calc,
    n,
    l,
    m,
    plane="xz",
    scale=None,
    grid_size=400,
    a0=None,
    scheme="classic",
    show_guides=True,
    show_labels=True,
    orbital_type="complex",
    real_component="cos",
    gamma=0.3,
    rel_eps=1e-6,
    vmax=None,
    background="white",
):
    """
    Static 2D probability-density map (|ψ|², with PowerNorm enhancement of low-density regions).

    Returns a matplotlib Figure (does not call show()).
    """
    a0 = a0 or calc.a0
    scheme_d = resolve_scheme(scheme)
    data = build_plane_data(
        calc, n, l, m, plane=plane, scale=scale, grid_size=grid_size,
        orbital_type=orbital_type, real_component=real_component,
    )
    d, _ = normalize_nonneg(data["density"], rel_eps=rel_eps, vmax=vmax)
    r_max = calc.classical_turning_points(n, l)[1]
    meta = plane_meta(plane)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=1.0)
    im = ax.imshow(
        d, extent=[-data["scale"], data["scale"], -data["scale"], data["scale"]],
        origin="lower", cmap=cmap_for(scheme, "density"), norm=norm,
        aspect="equal", interpolation="bilinear",
    )

    if show_guides:
        draw_contours(ax, data["C1"], data["C2"], d, show_labels=show_labels)
        draw_classical_circle(
            ax, r_max, color=scheme_d.get("boundary", "red"),
            label="Classical boundary",
        )
        if show_labels:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    if show_labels:
        ax.set_xlabel(data["xlabel"], fontsize=11)
        ax.set_ylabel(data["ylabel"], fontsize=11)
        ax.set_title(
            rf"Probability Density $|\psi|^2$ — "
            rf"n={n}, l={l}, m={m}  ({meta['name']} cross-section, {meta['note']})",
            fontsize=14, fontweight="bold", pad=18,
        )
        add_info_box(ax, n, l, m, r_max, a0, data["scale"])

    style_2d_axis(ax, background=background, show_labels=show_labels)
    if data["nodal_plane"]:
        add_nodal_note_2d(ax, background=background)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if show_labels:
        cbar.set_label(r"$|\psi|^2$ (relative, normalized to max = 1)")
    style_colorbar(cbar, background)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dynamic
# ---------------------------------------------------------------------------
def plot1_density_dynamic(
    calc,
    n,
    l,
    m,
    plane="xz",
    scale=None,
    grid_size=300,
    a0=None,
    scheme="classic",
    dt=2.0,
    frames=200,
    interval=30,
    show_labels=True,
    background="black",
    orbital_type="complex",
    real_component="cos",
):
    """
    Time-dependent 2D cross-section animation: a 1×3 panel of Re ψ, Im ψ, and |ψ|².

    Returns (animation, figure); the caller is responsible for plt.show() or ani.save().
    """
    data = build_plane_data(
        calc, n, l, m, plane=plane, scale=scale, grid_size=grid_size,
        orbital_type=orbital_type, real_component=real_component,
    )
    ani, fig = phase_animation(
        n=n, l=l, m=m,
        psi=data["psi"],
        third_data=data["density"],
        third_title=r"$|\psi|^2$",
        scheme=scheme,
        third_cmap=cmap_for(scheme, "density"),
        scale=data["scale"],
        plane=plane,
        dt=dt, frames=frames, interval=interval,
        show_labels=show_labels, background=background,
        third_vmin=0.0, third_vmax=1.0,
        nodal_note=data["nodal_note"],
    )
    return ani, fig



def plot1_density_dynamic_three_planes(
    calc,
    n,
    l,
    m,
    scale=None,
    grid_size=220,
    scheme="classic",
    dt=2.0,
    frames=120,
    interval=40,
    show_labels=True,
    background="black",
    orbital_type="complex",
    real_component="cos",
):
    """Dynamic 3x3 view: XY/XZ/YZ rows x Re/Im/density columns."""
    ani, fig = phase_animation_three_planes(
        calc, n, l, m, scale=scale, grid_size=grid_size, scheme=scheme,
        dt=dt, frames=frames, interval=interval, show_labels=show_labels,
        background=background, orbital_type=orbital_type,
        real_component=real_component,
    )
    return ani, fig
