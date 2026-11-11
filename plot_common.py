"""
plot_common.py — shared styles and helpers for 2D / 3D plotting
====================================================

Centralizes the contour, classical-boundary, info-box, and dark-axis styling
shared by the plot modules so the figures stay visually consistent.
"""

from __future__ import annotations

import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from palettes import cmap_for, is_magnitude


# ---------------------------------------------------------------------------
# Data normalization
# ---------------------------------------------------------------------------
def normalize_nonneg(data, rel_eps=1e-6, vmax=None):
    """
    Normalize a non-negative field (|psi|^2 or 4*pi*r^2*|psi|^2) to [0, 1] and suppress nodal residuals.

    Returns (normalized, vmax_used).
    """
    d = np.asarray(data, dtype=float).copy()
    if vmax is None:
        vmax = float(np.nanmax(d)) if d.size else 0.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    cutoff = vmax * float(rel_eps)
    d = np.where(d < cutoff, 0.0, d)
    d = d / vmax
    return np.clip(d, 0.0, 1.0), vmax


# ---------------------------------------------------------------------------
# 2D style
# ---------------------------------------------------------------------------
def style_2d_axis(ax, background="black", show_labels=True):
    """Axis, tick, spine, and label colors for dark/light backgrounds."""
    fg = "white" if str(background).lower() == "black" else "black"
    ax.set_facecolor(background)
    ax.grid(False)
    ax.tick_params(colors=fg, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(fg)
        spine.set_visible(show_labels)
    if not show_labels:
        ax.set_xticks([])
        ax.set_yticks([])
    for label in (ax.xaxis.label, ax.yaxis.label, ax.title):
        label.set_color(fg)
    use_scientific_axis(ax, ("x", "y"))
    return fg


def use_scientific_axis(ax, which=("x", "y", "z")):
    """Use mathtext scientific notation (e.g. 10^-9) to avoid collisions with the title."""
    from matplotlib.ticker import ScalarFormatter

    for name in which:
        axis = getattr(ax, f"{name}axis", None)
        if axis is None:
            continue
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 2))
        axis.set_major_formatter(fmt)
        try:
            axis.get_offset_text().set_fontsize(8)
        except Exception:
            pass


def draw_contours(ax, C1, C2, data, show_labels=True, n_levels=12):
    """Overlay logarithmically spaced contours (white lines with a dark outline) on a 2D field."""
    dmax = float(np.nanmax(data)) if np.size(data) else 0.0
    if not np.isfinite(dmax) or dmax <= 0:
        return None
    lo = max(dmax * 1e-6, 1e-12)
    levels = np.logspace(np.log10(lo), np.log10(dmax), n_levels)
    levels = np.unique(levels)
    if levels.size < 2:
        return None
    cs = ax.contour(
        C1, C2, data, levels=levels,
        colors="white", alpha=0.75, linewidths=0.8,
    )
    try:
        cs.set_path_effects(
            [pe.withStroke(linewidth=1.7, foreground="black", alpha=0.55)]
        )
    except Exception:
        pass
    if show_labels:
        try:
            ax.clabel(cs, inline=True, fontsize=7, fmt="%1.0e")
        except Exception:
            pass
    return cs


def draw_classical_circle(ax, r_max, color="red", label="Classical boundary"):
    """Draw the classical turning-point circle (r = r_max on a 2D cross-section)."""
    theta = np.linspace(0.0, 2.0 * np.pi, 360)
    x = r_max * np.cos(theta)
    y = r_max * np.sin(theta)
    outline = "black" if str(color).lower() in ("white", "yellow") else "white"
    line, = ax.plot(
        x, y, color=color, ls="--", lw=2.2, alpha=0.9,
        label=label,
    )
    line.set_path_effects(
        [pe.withStroke(linewidth=3.4, foreground=outline, alpha=0.55)]
    )
    return line


def add_info_box(ax, n, l, m, r_max, a0, scale):
    """Top-left info box: quantum numbers, classical radius, Bohr radius, and display range."""
    info = (
        f"n = {n},  l = {l},  m = {m}\n"
        f"classical r_max = {r_max / a0:.1f} a0\n"
        f"Bohr a0 = {a0:.3e} m\n"
        f"capture range = {scale / a0:.1f} a0"
    )
    ax.text(
        0.02, 0.98, info,
        transform=ax.transAxes, va="top", ha="left",
        fontsize=10, color="black",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9,
                  edgecolor="0.7"),
    )


def style_colorbar(cbar, background="white"):
    """Apply consistent colorbar ticks, labels, and background for dark/light themes."""
    fg = "white" if str(background).lower() == "black" else "black"
    cbar.ax.tick_params(colors=fg)
    try:
        cbar.ax.yaxis.label.set_color(fg)
    except Exception:
        pass
    try:
        cbar.outline.set_edgecolor(fg)
    except Exception:
        pass
    cbar.ax.set_facecolor(background)


# ---------------------------------------------------------------------------
# 3D style
# ---------------------------------------------------------------------------
def style_3d_axis(
    ax,
    background="black",
    show_labels=True,
    scale=None,
    xlabel="X (m)",
    ylabel="Z (m)",
    zlabel="",
):
    """Apply a consistent 3D axis palette; the background may be black or white."""
    bg = "black" if str(background).lower() == "black" else "white"
    fg = "white" if bg == "black" else "black"
    pane = (0, 0, 0, 1) if bg == "black" else (1, 1, 1, 1)

    ax.set_facecolor(bg)
    ax.tick_params(colors=fg, labelsize=8)
    ax.grid(False)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane)
        axis.pane.set_edgecolor(fg)
        axis.line.set_color(fg)
        axis.label.set_color(fg)

    if scale is not None:
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
    if show_labels:
        ax.set_xlabel(xlabel, color=fg, fontsize=9)
        ax.set_ylabel(ylabel, color=fg, fontsize=9)
        ax.set_zlabel(zlabel, color=fg, fontsize=9)
    use_scientific_axis(ax, ("x", "y", "z"))


def set_3d_view(ax, elev=28, azim=-60):
    """A consistent 3D viewing angle."""
    ax.view_init(elev=elev, azim=azim)


def _nodal_box(background):
    fg = "white" if str(background).lower() == "black" else "black"
    return dict(
        boxstyle="round", facecolor=background, alpha=0.88,
        edgecolor=fg,
    )


def add_nodal_note_2d(ax, background="black",
                      text="Nodal plane: |ψ| = 0 on this section"):
    """Mark a nodal plane on a 2D section so a blank panel is not mistaken for a rendering failure."""
    fg = "white" if str(background).lower() == "black" else "black"
    ax.text(
        0.5, 0.5, text, transform=ax.transAxes,
        ha="center", va="center", color=fg, fontsize=11,
        bbox=_nodal_box(background),
    )


def add_nodal_note_3d(ax, background="black",
                      text="Nodal plane: |ψ| = 0 on this section"):
    """Mark a nodal plane on a 3D section."""
    fg = "white" if str(background).lower() == "black" else "black"
    ax.text2D(
        0.5, 0.5, text, transform=ax.transAxes,
        ha="center", va="center", color=fg, fontsize=11,
        bbox=_nodal_box(background),
    )


def add_nodal_note_fig(fig, background="black",
                       text="Nodal plane: |ψ| = 0 on this section"):
    """Mark a nodal plane on the whole figure (for the dynamic three-panel view)."""
    fg = "white" if str(background).lower() == "black" else "black"
    fig.text(
        0.5, 0.5, text, ha="center", va="center", color=fg, fontsize=12,
        bbox=_nodal_box(background),
    )


def plane_axis_labels(plane, dynamic=False):
    """Return (xlabel, ylabel) for a plane."""
    meta = {
        "xy": ("x (m)", "y (m)"),
        "xz": ("x (m)", "z (m)"),
        "yz": ("y (m)", "z (m)"),
    }
    key = str(plane).lower()
    return meta.get(key, meta["xz"])


def plane_axis_labels_static(plane):
    """Axis labels for static figures (capitalized)."""
    meta = {
        "xy": ("X (m)", "Y (m)"),
        "xz": ("X (m)", "Z (m)"),
        "yz": ("Y (m)", "Z (m)"),
    }
    key = str(plane).lower()
    return meta.get(key, meta["xz"])


def surface_stride(grid_size, res_factor=1.0, max_points=110):
    """
    Downsampling step for 3D surfaces.

    res_factor=1.0 uses the full grid (capped at max_points x max_points
    to keep a 2000 x 2000 grid from freezing the 3D surface); res_factor=0.2 keeps about 20% of
    the samples.
    """
    res_factor = max(float(res_factor), 1e-3)
    stride = max(1, int(round(1.0 / res_factor)))
    floor = max(1, int(np.ceil(grid_size / float(max_points))))
    return max(stride, floor)


# ---------------------------------------------------------------------------
# Time-dependent 2D three-panel view (Re / Im / target) shared by plot1 and plot2
# ---------------------------------------------------------------------------
def phase_animation(
    n,
    l,
    m,
    psi,
    third_data,
    third_title,
    scheme,
    third_cmap,
    scale,
    plane="xz",
    dt=2.0,
    frames=200,
    interval=30,
    show_labels=True,
    background="black",
    third_vmin=0.0,
    third_vmax=1.0,
    nodal_note="",
):
    """
    Build a 1x3 phase-evolution animation: Re psi, Im psi, and a third panel (|psi|^2 or 4*pi*r^2*|psi|^2).

    psi must be a 2D complex array normalized by max|psi|.
    Returns (animation, figure).
    """
    mag = is_magnitude(scheme)
    psi_real = np.real(psi)
    psi_imag = np.imag(psi)
    psi_abs_max = float(np.max(np.abs(psi))) if psi.size else 0.0

    # Stationary-state phase rotation makes Re/Im oscillate between 0 and max|psi|;
    # use max|psi| for the fixed color scale, not max|Re psi| / max|Im psi| at t=0 (which may be 0).
    if mag:
        data_r = np.abs(psi_real)
        data_i = np.abs(psi_imag)
        vmax_r = vmax_i = max(psi_abs_max, 1e-12)
        vmin_r = vmin_i = 0.0
        title_r, title_i = "|Re ψ|", "|Im ψ|"
    else:
        data_r = psi_real
        data_i = psi_imag
        vmax = max(psi_abs_max, 1e-12)
        vmin_r, vmax_r = -vmax, vmax
        vmin_i, vmax_i = -vmax, vmax
        title_r, title_i = "Re ψ", "Im ψ"

    cmap_r = cmap_for(scheme, "real")
    cmap_i = cmap_for(scheme, "imag")
    fg = "white" if str(background).lower() == "black" else "black"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(background)
    extent = [-scale, scale, -scale, scale]
    xlabel, ylabel = plane_axis_labels(plane, dynamic=True)

    specs = [
        (axes[0], data_r, cmap_r, vmin_r, vmax_r, title_r),
        (axes[1], data_i, cmap_i, vmin_i, vmax_i, title_i),
        (axes[2], third_data, third_cmap, third_vmin, third_vmax, third_title),
    ]
    ims = []
    for ax, arr, cmap, vmin, vmax, title in specs:
        plot_arr = np.ma.masked_all(np.shape(arr)) if nodal_note else arr
        im = ax.imshow(
            plot_arr, extent=extent, origin="lower", cmap=cmap,
            vmin=vmin, vmax=vmax, aspect="equal", interpolation="bilinear",
        )
        ims.append(im)
        style_2d_axis(ax, background=background, show_labels=show_labels)
        ax.set_title(title, fontsize=12, color=fg)
        if show_labels:
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)

    if show_labels:
        fig.suptitle(
            rf"n={n}, l={l}, m={m}  ({plane.upper()} cross-section) — phase evolution",
            color=fg, fontsize=15,
        )

    time_text = axes[0].text(
        0.02, 0.98, "", transform=axes[0].transAxes,
        color="cyan" if str(background).lower() == "black" else "darkblue",
        fontsize=12, va="top",
        bbox=dict(boxstyle="round", facecolor=background, alpha=0.75),
    )
    if nodal_note:
        add_nodal_note_fig(fig, background=background, text=nodal_note)

    e_n = -0.5 / (n * n)

    def update(frame):
        t = frame * dt
        if nodal_note:
            time_text.set_text(f"t = {t:.1f} a.u.")
            return ims + [time_text]
        psi_t = psi * np.exp(-1j * e_n * t)
        re_t = np.real(psi_t)
        im_t = np.imag(psi_t)
        if mag:
            ims[0].set_data(np.abs(re_t))
            ims[1].set_data(np.abs(im_t))
        else:
            ims[0].set_data(re_t)
            ims[1].set_data(im_t)
        ims[2].set_data(third_data)  # the stationary |psi|^2 / 4*pi*r^2*|psi|^2 is time-independent
        time_text.set_text(f"t = {t:.1f} a.u.")
        return ims + [time_text]

    ani = FuncAnimation(
        fig, update, frames=int(max(1, frames)), interval=interval, blit=False,
    )
    fig.tight_layout()
    return ani, fig


def phase_animation_three_planes(
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
    """3x3 dynamic view: XY/XZ/YZ rows x Re/Im/density columns."""
    from core import build_plane_data

    planes = [("xy", "XY", "z=0"), ("xz", "XZ", "y=0"), ("yz", "YZ", "x=0")]
    data = []
    global_abs = 0.0
    for p, _, _ in planes:
        d = build_plane_data(
            calc, n, l, m, plane=p, scale=scale, grid_size=grid_size,
            orbital_type=orbital_type, real_component=real_component,
        )
        data.append(d)
        if d["psi"].size:
            global_abs = max(global_abs, float(np.max(np.abs(d["psi"]))))

    vmax = max(global_abs, 1e-12)
    mag = is_magnitude(scheme)
    cmap_r = cmap_for(scheme, "real")
    cmap_i = cmap_for(scheme, "imag")
    cmap_d = cmap_for(scheme, "density")
    fg = "white" if str(background).lower() == "black" else "black"

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.patch.set_facecolor(background)
    ims = []

    if mag:
        col_titles = [r"$|\mathrm{Re}\,\psi|$", r"$|\mathrm{Im}\,\psi|$", r"$|\psi|^2$"]
    else:
        col_titles = [r"$\mathrm{Re}\,\psi$", r"$\mathrm{Im}\,\psi$", r"$|\psi|^2$"]

    for i, (p, name, note) in enumerate(planes):
        d = data[i]
        ext = [-d["scale"], d["scale"], -d["scale"], d["scale"]]
        nodal = d["nodal_plane"]
        psi = d["psi"]
        if mag:
            arrs = [np.abs(np.real(psi)), np.abs(np.imag(psi)), d["density"]]
        else:
            arrs = [np.real(psi), np.imag(psi), d["density"]]

        for j in range(3):
            ax = axes[i, j]
            style_2d_axis(ax, background=background, show_labels=show_labels)
            if j == 2:
                vmin, vmax_j = 0.0, 1.0
            elif mag:
                vmin, vmax_j = 0.0, vmax
            else:
                vmin, vmax_j = -vmax, vmax

            if nodal:
                # Keep the axes dark and draw an invisible image for the
                # animation update; this avoids a white masked-array panel.
                ax.set_facecolor(background)
                im = ax.imshow(
                    np.zeros_like(arrs[j]), extent=ext, origin="lower",
                    cmap=[cmap_r, cmap_i, cmap_d][j],
                    vmin=vmin, vmax=vmax_j, aspect="equal",
                    alpha=0.0,
                )
            else:
                im = ax.imshow(
                    arrs[j], extent=ext, origin="lower",
                    cmap=[cmap_r, cmap_i, cmap_d][j],
                    vmin=vmin, vmax=vmax_j, aspect="equal",
                    interpolation="bilinear",
                )
            ims.append(im)

            if i == 0:
                ax.set_title(col_titles[j], fontsize=12, color=fg)
            xl, yl = plane_axis_labels(p, dynamic=True)
            if show_labels:
                ax.set_xlabel(xl, fontsize=9, color=fg)
                ax.set_ylabel(yl, fontsize=9, color=fg)
            if nodal:
                ax.text(
                    0.5, 0.5, "Nodal plane", transform=ax.transAxes,
                    ha="center", va="center", color=fg, fontsize=10,
                    alpha=0.85,
                )

    if show_labels:
        fig.suptitle(
            rf"n={n}, l={l}, m={m}  (three planes, dynamic)",
            color=fg, fontsize=16,
        )

    time_text = axes[0, 0].text(
        0.02, 0.98, "", transform=axes[0, 0].transAxes,
        color="cyan" if str(background).lower() == "black" else "darkblue",
        fontsize=12, va="top",
        bbox=dict(boxstyle="round", facecolor=background, alpha=0.75),
    )

    e_n = -0.5 / (n * n)

    def update(frame):
        t = frame * dt
        for i, d in enumerate(data):
            psi_t = d["psi"] * np.exp(-1j * e_n * t)
            re_t = np.real(psi_t)
            im_t = np.imag(psi_t)
            if mag:
                ims[3 * i].set_data(np.abs(re_t))
                ims[3 * i + 1].set_data(np.abs(im_t))
            else:
                ims[3 * i].set_data(re_t)
                ims[3 * i + 1].set_data(im_t)
            ims[3 * i + 2].set_data(d["density"])
        time_text.set_text(f"t = {t:.1f} a.u.")
        return ims + [time_text]

    ani = FuncAnimation(
        fig, update, frames=int(max(1, frames)), interval=interval, blit=False,
    )
    fig.tight_layout()
    fig.subplots_adjust(left=0.14)
    for i, (p, name, note) in enumerate(planes):
        pos = axes[i, 0].get_position()
        fig.text(
            0.006, (pos.y0 + pos.y1) / 2,
            f"{name}\n({note})", ha="left", va="center",
            rotation=0, color=fg, fontsize=10,
        )
    return ani, fig
