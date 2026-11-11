"""
gui_main.py — Time_Hydrogen_v2 GUI
========================================

Following the New ICS model, the six plot types are arranged in a 3x2 matrix:

                    probability |psi|^2        shell density 4*pi*r^2*|psi|^2
    2D cross-section  plot1                     plot2
    3D surface (height field)  plot3           plot4
    3D volume rendering        plot5 (Plotly)  plot6 (Plotly)

Each cell has "Static" and "Dynamic" buttons; a separate entry opens the standalone 3D point cloud.

Colors
----
The dropdown provides eight schemes (classic, red/blue diverging, general colorblind-friendly,
green/red/blue deficiencies, grayscale, high-saturation), with previews and CVD notes; custom matplotlib colormaps are also supported.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

import tkinter as tk
from tkinter import ttk, messagebox

from core import HydrogenWavefunction
from palettes import (
    SCHEME_KEYS,
    cmap_for,
    make_custom_scheme,
    resolve_scheme,
)
import plot1
import plot2
import plot3
import plot4
import plot5
import plot6
import pointcloud_static

plt.rcParams["font.family"] = ["Times New Roman", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


class HydrogenGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Time-Dependent Hydrogen Wavefunction Visualizer v2 — Time_Hydrogen_v2")
        self.root.geometry("1200x800")
        self.root.minsize(1100, 720)

        self.calc = None
        self.n = self.l = self.m = None
        self.grid_size = None
        self.orbital = None
        self.real_component = None
        self.scale = None
        self.custom_scheme = None
        self._figs = []
        self._anim = None

        self._build_widgets()
        self._on_scheme_change()

    # ==================================================================
    # Interface
    # ==================================================================
    def _build_widgets(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Header.TLabel", font=("Microsoft YaHei UI", 14, "bold"))
        style.configure("Hint.TLabel", foreground="#555555", font=("Microsoft YaHei UI", 9))

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            outer, text="Time-Dependent Hydrogen Wavefunction Visualizer v2", style="Header.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            outer,
            text="3x2 plot matrix: 2D cross-section / 3D surface (height field) / 3D volume rendering x "
                 "probability |psi|^2 / shell density 4*pi*r^2*|psi|^2",
            style="Hint.TLabel",
        ).pack(anchor=tk.W, pady=(0, 6))

        self._build_quantum(outer)
        self._build_display(outer)
        self._build_animation(outer)
        self._build_plot_matrix(outer)

        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN,
            anchor=tk.W, padding=(6, 2),
        )
        status.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------
    def _build_quantum(self, parent):
        frame = ttk.LabelFrame(parent, text="1. Quantum numbers and grid", padding=8)
        frame.pack(fill=tk.X, pady=4)

        ttk.Label(frame, text="n:").grid(row=0, column=0, sticky=tk.W)
        self.n_var = tk.StringVar(value="3")
        ttk.Entry(frame, textvariable=self.n_var, width=6).grid(
            row=0, column=1, padx=(2, 10))

        ttk.Label(frame, text="l:").grid(row=0, column=2, sticky=tk.W)
        self.l_var = tk.StringVar(value="2")
        ttk.Entry(frame, textvariable=self.l_var, width=6).grid(
            row=0, column=3, padx=(2, 10))

        ttk.Label(frame, text="m:").grid(row=0, column=4, sticky=tk.W)
        self.m_var = tk.StringVar(value="1")
        ttk.Entry(frame, textvariable=self.m_var, width=6).grid(
            row=0, column=5, padx=(2, 10))

        ttk.Label(frame, text="Grid:").grid(row=0, column=6, sticky=tk.W)
        self.grid_var = tk.StringVar(value="400")
        ttk.Entry(frame, textvariable=self.grid_var, width=7).grid(
            row=0, column=7, padx=(2, 10))

        ttk.Label(frame, text="Plane:").grid(row=0, column=8, sticky=tk.W)
        self.plane_var = tk.StringVar(value="xz")
        ttk.Combobox(
            frame, textvariable=self.plane_var, values=["xz", "xy", "yz"],
            state="readonly", width=5,
        ).grid(row=0, column=9, padx=(2, 10))

        self.three_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame, text="3x3 three-plane (dynamic)", variable=self.three_var,
        ).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))

        ttk.Label(frame, text="Orbital:").grid(row=0, column=10, sticky=tk.W)
        self.orbital_var = tk.StringVar(value="complex")
        ttk.Combobox(
            frame, textvariable=self.orbital_var, values=["complex", "real"],
            state="readonly", width=8,
        ).grid(row=0, column=11, padx=(2, 10))

        ttk.Label(frame, text="Real part:").grid(row=0, column=12, sticky=tk.W)
        self.real_comp_var = tk.StringVar(value="cos")
        ttk.Combobox(
            frame, textvariable=self.real_comp_var, values=["cos", "sin"],
            state="readonly", width=5,
        ).grid(row=0, column=13, padx=(2, 0))

    # ------------------------------------------------------------------
    def _build_display(self, parent):
        frame = ttk.LabelFrame(parent, text="2. Display and colors (colorblind-friendly)", padding=8)
        frame.pack(fill=tk.X, pady=4)

        self.guides_var = tk.BooleanVar(value=True)
        self.labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Guides / contours", variable=self.guides_var).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 8))
        ttk.Checkbutton(frame, text="Titles and axis labels", variable=self.labels_var).grid(
            row=0, column=1, sticky=tk.W, padx=(0, 8))

        ttk.Label(frame, text="Background:").grid(row=0, column=2, sticky=tk.W)
        self.bg_var = tk.StringVar(value="black")
        ttk.Combobox(
            frame, textvariable=self.bg_var, values=["black", "white"],
            state="readonly", width=7,
        ).grid(row=0, column=3, padx=(2, 12))

        ttk.Label(frame, text="Scheme:").grid(row=0, column=4, sticky=tk.W)
        self.scheme_var = tk.StringVar(value="classic")
        self.scheme_combo = ttk.Combobox(
            frame, textvariable=self.scheme_var,
            values=list(SCHEME_KEYS), state="readonly", width=14,
        )
        self.scheme_combo.grid(row=0, column=5, padx=(2, 6))
        self.scheme_combo.bind("<<ComboboxSelected>>",
                               lambda _e: self._on_scheme_change())

        ttk.Button(frame, text="Custom colors...", command=self._open_custom_dialog).grid(
            row=0, column=6, padx=(2, 0))

        self.scheme_desc = ttk.Label(
            frame, text="", style="Hint.TLabel", justify=tk.LEFT, wraplength=980,
        )
        self.scheme_desc.grid(row=1, column=0, columnspan=7, sticky=tk.W,
                              pady=(4, 2))

        self.preview = tk.Canvas(
            frame, height=56, bg="white", highlightthickness=1,
            highlightbackground="#888888",
        )
        self.preview.grid(row=2, column=0, columnspan=7, sticky=tk.EW, pady=(2, 0))
        self.preview.bind("<Configure>", lambda _e: self._draw_preview())

    # ------------------------------------------------------------------
    def _build_animation(self, parent):
        frame = ttk.LabelFrame(
            parent, text="3. Animation and volume parameters", padding=8)
        frame.pack(fill=tk.X, pady=4)

        ttk.Label(frame, text="dt:").grid(row=0, column=0, sticky=tk.W)
        self.dt_var = tk.StringVar(value="2.0")
        ttk.Entry(frame, textvariable=self.dt_var, width=6).grid(
            row=0, column=1, padx=(2, 10))

        ttk.Label(frame, text="Frames:").grid(row=0, column=2, sticky=tk.W)
        self.frames_var = tk.StringVar(value="160")
        ttk.Entry(frame, textvariable=self.frames_var, width=6).grid(
            row=0, column=3, padx=(2, 10))

        ttk.Label(frame, text="Interval ms:").grid(row=0, column=4, sticky=tk.W)
        self.interval_var = tk.StringVar(value="40")
        ttk.Entry(frame, textvariable=self.interval_var, width=6).grid(
            row=0, column=5, padx=(2, 10))

        ttk.Label(frame, text="Surface sampling:").grid(row=0, column=6, sticky=tk.W)
        self.res_factor_var = tk.StringVar(value="0.2")
        ttk.Entry(frame, textvariable=self.res_factor_var, width=6).grid(
            row=0, column=7, padx=(2, 10))
        ttk.Label(frame, text="(0.05-1.0; larger is finer)").grid(
            row=0, column=8, sticky=tk.W)

        ttk.Label(frame, text="Volume resolution:").grid(row=1, column=0, sticky=tk.W,
                                                   pady=(4, 0))
        self.volume_res_var = tk.StringVar(value="50")
        ttk.Entry(frame, textvariable=self.volume_res_var, width=6).grid(
            row=1, column=1, padx=(2, 10), pady=(4, 0))

        ttk.Label(frame, text="σ:").grid(row=1, column=2, sticky=tk.W, pady=(4, 0))
        self.sigma_var = tk.StringVar(value="1.0")
        ttk.Entry(frame, textvariable=self.sigma_var, width=6).grid(
            row=1, column=3, padx=(2, 10), pady=(4, 0))

        ttk.Label(frame, text="Isosurfaces:").grid(row=1, column=4, sticky=tk.W,
                                                pady=(4, 0))
        self.surface_var = tk.StringVar(value="50")
        ttk.Entry(frame, textvariable=self.surface_var, width=6).grid(
            row=1, column=5, padx=(2, 10), pady=(4, 0))

        ttk.Label(frame, text="Opacity:").grid(row=1, column=6, sticky=tk.W,
                                                pady=(4, 0))
        self.opacity_var = tk.StringVar(value="0.10")
        ttk.Entry(frame, textvariable=self.opacity_var, width=6).grid(
            row=1, column=7, padx=(2, 10), pady=(4, 0))

        ttk.Label(frame, text="Volume frames:").grid(row=1, column=8, sticky=tk.W,
                                                  pady=(4, 0))
        self.volume_frames_var = tk.StringVar(value="40")
        ttk.Entry(frame, textvariable=self.volume_frames_var, width=6).grid(
            row=1, column=9, padx=(2, 0), pady=(4, 0))

        ttk.Label(frame, text="Volume component:").grid(row=2, column=0, sticky=tk.W,
                                                  pady=(4, 0))
        self.comp_var = tk.StringVar(value="both")
        for i, (text, value) in enumerate(
                [("Real", "real"), ("Imag", "imag"), ("Real+Imag", "both")]):
            ttk.Radiobutton(
                frame, text=text, variable=self.comp_var, value=value,
            ).grid(row=2, column=1 + i, sticky=tk.W, pady=(4, 0))

    # ------------------------------------------------------------------
    def _build_plot_matrix(self, parent):
        frame = ttk.LabelFrame(parent, text="4. Plot matrix (each cell: Static / Dynamic)",
                               padding=8)
        frame.pack(fill=tk.X, pady=4)

        for col, text in enumerate(
                ["", "probability |psi|^2", "shell density 4*pi*r^2*|psi|^2"]):
            ttk.Label(frame, text=text, font=("Helvetica", 10, "bold")).grid(
                row=0, column=col, padx=6, pady=(0, 4), sticky=tk.W)

        rows = [
            ("2D cross-section", 1, 2),
            ("3D surface (height field)", 3, 4),
            ("3D volume rendering", 5, 6),
        ]
        for r, (label, left_id, right_id) in enumerate(rows, start=1):
            ttk.Label(frame, text=label, font=("Helvetica", 10, "bold")).grid(
                row=r, column=0, sticky=tk.W, padx=6, pady=3)
            for c, plot_id in enumerate((left_id, right_id), start=1):
                cell = ttk.Frame(frame)
                cell.grid(row=r, column=c, sticky=tk.W, padx=4, pady=3)
                ttk.Button(
                    cell, text="Static", width=10,
                    command=lambda p=plot_id: self._run(p, "s"),
                ).pack(side=tk.LEFT, padx=2)
                ttk.Button(
                    cell, text="Dynamic", width=10,
                    command=lambda p=plot_id: self._run(p, "d"),
                ).pack(side=tk.LEFT, padx=2)

        extra = ttk.Frame(frame)
        extra.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=6, pady=(8, 0))
        ttk.Button(
            extra, text="Standalone static 3D point cloud (single-file tool)...", width=34,
            command=lambda: self._run(7, "s"),
        ).pack(side=tk.LEFT)
        ttk.Label(
            extra,
            text="  Runs pointcloud_static.py; can also be used from the command line",
            style="Hint.TLabel",
        ).pack(side=tk.LEFT)

    # ==================================================================
    # Colors
    # ==================================================================
    def _current_scheme(self):
        key = self.scheme_var.get()
        if key == "custom":
            if self.custom_scheme is None:
                self._open_custom_dialog()
            return self.custom_scheme or "classic"
        return key

    def _on_scheme_change(self):
        scheme = self._current_scheme()
        try:
            s = resolve_scheme(scheme)
            mode = "magnitude" if s.get("magnitude") else "signed"
            text = (
                f"{s['label']}  |  {s['cvd']}  |  Re/Im: {mode} mode\n"
                f"{s['description']}"
            )
        except Exception:
            text = "Invalid color scheme."
        self.scheme_desc.config(text=text)
        self._draw_preview()

    def _draw_preview(self):
        canvas = getattr(self, "preview", None)
        if canvas is None:
            return
        canvas.delete("all")
        width = max(canvas.winfo_width(), 640)
        scheme = self._current_scheme()
        try:
            roles = [
                ("Density", cmap_for(scheme, "density")),
                ("Real", cmap_for(scheme, "real")),
                ("Imag", cmap_for(scheme, "imag")),
            ]
        except Exception:
            return

        x0 = 58
        x1 = max(width - 10, x0 + 60)
        for row, (label, cmap) in enumerate(roles):
            y = 5 + row * 19
            canvas.create_text(6, y + 7, text=label, anchor=tk.W,
                               fill="#222222", font=("Microsoft YaHei UI", 8))
            n_seg = 64
            for i in range(n_seg):
                xa = x0 + (x1 - x0) * i / n_seg
                xb = x0 + (x1 - x0) * (i + 1) / n_seg
                canvas.create_rectangle(
                    xa, y, xb, y + 14, outline="", fill=to_hex(cmap(i / (n_seg - 1))),
                )
            canvas.create_rectangle(x0, y, x1, y + 14, outline="#444444")

    def _open_custom_dialog(self):
        top = tk.Toplevel(self.root)
        top.title("Custom color scheme")
        top.transient(self.root)
        top.grab_set()
        top.resizable(False, False)

        entries = {}
        fields = [
            ("density", "Density |psi|^2 colormap:", "viridis"),
            ("shell", "Shell-density colormap:", "inferno"),
            ("real", "Real-part colormap:", "PuOr"),
            ("imag", "Imaginary-part colormap:", "PuOr"),
            ("signed", "Diverging colormap:", "PuOr"),
        ]
        for r, (key, label, default) in enumerate(fields):
            ttk.Label(top, text=label).grid(row=r, column=0, sticky=tk.W,
                                            padx=8, pady=4)
            var = tk.StringVar(value=default)
            ttk.Entry(top, textvariable=var, width=16).grid(
                row=r, column=1, padx=8, pady=4)
            entries[key] = var

        magnitude = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top, text="Re/Im use magnitude mode (|Re|, |Im|; 0 at the dark end)",
            variable=magnitude,
        ).grid(row=len(fields), column=0, columnspan=2, sticky=tk.W,
               padx=8, pady=(4, 2))

        def on_ok():
            try:
                scheme = make_custom_scheme(
                    density=entries["density"].get().strip() or "viridis",
                    shell=entries["shell"].get().strip() or "inferno",
                    real=entries["real"].get().strip() or "PuOr",
                    imag=entries["imag"].get().strip() or "PuOr",
                    signed=entries["signed"].get().strip() or "PuOr",
                    magnitude=magnitude.get(),
                )
            except KeyError as exc:
                messagebox.showerror("Colormap error", str(exc), parent=top)
                return
            self.custom_scheme = scheme
            values = list(SCHEME_KEYS) + ["custom"]
            self.scheme_combo["values"] = values
            self.scheme_var.set("custom")
            self._on_scheme_change()
            top.destroy()

        btns = ttk.Frame(top)
        btns.grid(row=len(fields) + 1, column=0, columnspan=2, pady=8)
        ttk.Button(btns, text="OK", command=on_ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Cancel", command=top.destroy).pack(side=tk.LEFT, padx=4)

    # ==================================================================
    # Parameter parsing
    # ==================================================================
    def _read_quantum(self):
        try:
            n = int(self.n_var.get())
            l = int(self.l_var.get())
            m = int(self.m_var.get())
            grid = int(self.grid_var.get())
        except ValueError:
            messagebox.showerror("Input error", "n, l, m, and grid must be integers.")
            return None
        if n <= 0:
            messagebox.showerror("Input error", "n must be a positive integer.")
            return None
        if not (0 <= l < n):
            messagebox.showerror("Input error", f"l must satisfy 0 <= l < {n}.")
            return None
        if not (-l <= m <= l):
            messagebox.showerror("Input error", f"m must satisfy -{l} <= m <= {l}.")
            return None
        if not (50 <= grid <= 2000):
            messagebox.showerror("Input error", "Grid must be between 50 and 2000.")
            return None
        return n, l, m, grid

    def _read_anim(self):
        try:
            dt = float(self.dt_var.get())
            frames = int(self.frames_var.get())
            interval = int(self.interval_var.get())
            res_factor = float(self.res_factor_var.get())
            volume_res = int(self.volume_res_var.get())
            sigma = float(self.sigma_var.get())
            surface_count = int(self.surface_var.get())
            opacity = float(self.opacity_var.get())
            volume_frames = int(self.volume_frames_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Animation/volume parameters must be numeric.")
            return None
        if dt <= 0 or frames < 2 or interval < 10:
            messagebox.showerror("Input error", "Require dt > 0, frames >= 2, interval >= 10.")
            return None
        if not (0.02 <= res_factor <= 1.0):
            messagebox.showerror("Input error", "Surface sampling must be between 0.02 and 1.0.")
            return None
        if not (20 <= volume_res <= 120 and 5 <= surface_count <= 120):
            messagebox.showerror("Input error", "Resolution 20-120, isosurfaces 5-120.")
            return None
        if not (0.0 <= sigma <= 5.0 and 0.01 <= opacity <= 1.0):
            messagebox.showerror("Input error", "Sigma 0-5, opacity 0.01-1.")
            return None
        if not (2 <= volume_frames <= 300):
            messagebox.showerror("Input error", "Volume frames must be between 2 and 300.")
            return None
        return dict(
            dt=dt, frames=frames, interval=interval, res_factor=res_factor,
            volume_res=volume_res, sigma=sigma,
            surface_count=surface_count, opacity=opacity,
            volume_frames=volume_frames, component=self.comp_var.get(),
        )

    def _ensure_calc(self):
        q = self._read_quantum()
        anim = self._read_anim()
        if q is None or anim is None:
            return None
        n, l, m, grid = q
        orbital = self.orbital_var.get()
        real_component = self.real_comp_var.get() if orbital == "real" else "cos"

        if (self.calc is not None and self.n == n and self.l == l
                and self.m == m and self.grid_size == grid
                and self.orbital == orbital
                and self.real_component == real_component):
            return self.calc

        calc = HydrogenWavefunction()
        self.calc = calc
        self.n, self.l, self.m = n, l, m
        self.grid_size = grid
        self.orbital = orbital
        self.real_component = real_component
        self.scale = calc.scale_for(n, l)
        self.status_var.set(
            f"Set n={n}, l={l}, m={m}, grid={grid}, {orbital}"
            + (f"/{real_component}" if orbital == "real" else "")
        )
        return calc

    # ==================================================================
    # Plot dispatch
    # ==================================================================
    def _run(self, plot_id, mode):
        calc = self._ensure_calc()
        if calc is None:
            return
        anim = self._read_anim()
        if anim is None:
            return
        scheme = self._current_scheme()
        background = self.bg_var.get()
        show_labels = self.labels_var.get()
        show_guides = self.guides_var.get()
        plane = self.plane_var.get()

        self.status_var.set(f"Generating plot{plot_id} ({'static' if mode == 's' else 'dynamic'}) ...")
        self.root.config(cursor="watch")
        self.root.update()
        try:
            n, l, m, grid = self.n, self.l, self.m, self.grid_size
            if plot_id == 1:
                if mode == "s":
                    fig = plot1.plot1_density_2d(
                        calc, n, l, m, plane=plane, scale=self.scale,
                        grid_size=grid, scheme=scheme,
                        show_guides=show_guides, show_labels=show_labels,
                        background=background,
                        orbital_type=self.orbital,
                        real_component=self.real_component,
                    )
                else:
                    if self.three_var.get():
                        ani, fig = plot1.plot1_density_dynamic_three_planes(
                            calc, n, l, m, scale=self.scale,
                            grid_size=grid, scheme=scheme,
                            dt=anim["dt"], frames=anim["frames"],
                            interval=anim["interval"], show_labels=show_labels,
                            background=background,
                            orbital_type=self.orbital,
                            real_component=self.real_component,
                        )
                    else:
                        ani, fig = plot1.plot1_density_dynamic(
                            calc, n, l, m, plane=plane, scale=self.scale,
                            grid_size=grid, scheme=scheme,
                            dt=anim["dt"], frames=anim["frames"],
                            interval=anim["interval"], show_labels=show_labels,
                            background=background,
                            orbital_type=self.orbital,
                            real_component=self.real_component,
                        )
                    self._anim = ani

            elif plot_id == 2:
                if mode == "s":
                    fig = plot2.plot2_shell_2d(
                        calc, n, l, m, plane=plane, scale=self.scale,
                        grid_size=grid, scheme=scheme,
                        show_guides=show_guides, show_labels=show_labels,
                        background=background,
                        orbital_type=self.orbital,
                        real_component=self.real_component,
                    )
                else:
                    ani, fig = plot2.plot2_shell_dynamic(
                        calc, n, l, m, plane=plane, scale=self.scale,
                        grid_size=grid, scheme=scheme,
                        dt=anim["dt"], frames=anim["frames"],
                        interval=anim["interval"], show_labels=show_labels,
                        background=background,
                        orbital_type=self.orbital,
                        real_component=self.real_component,
                    )
                    self._anim = ani

            elif plot_id == 3:
                if mode == "s":
                    fig = plot3.plot3_surface_density(
                        calc, n, l, m, plane=plane, scale=self.scale,
                        grid_size=grid, scheme=scheme,
                        show_guides=show_guides, show_labels=show_labels,
                        background=background,
                        orbital_type=self.orbital,
                        real_component=self.real_component,
                    )
                else:
                    ani, fig = plot3.plot3_surface_density_dynamic(
                        calc, n, l, m, plane=plane, scale=self.scale,
                        grid_size=grid, scheme=scheme,
                        dt=anim["dt"], frames=anim["frames"],
                        interval=anim["interval"], show_labels=show_labels,
                        background=background, res_factor=anim["res_factor"],
                        orbital_type=self.orbital,
                        real_component=self.real_component,
                    )
                    self._anim = ani

            elif plot_id == 4:
                if mode == "s":
                    fig = plot4.plot4_surface_shell(
                        calc, n, l, m, plane=plane, scale=self.scale,
                        grid_size=grid, scheme=scheme,
                        show_box=show_guides, show_labels=show_labels,
                        background=background,
                        orbital_type=self.orbital,
                        real_component=self.real_component,
                    )
                else:
                    ani, fig = plot4.plot4_surface_shell_dynamic(
                        calc, n, l, m, plane=plane, scale=self.scale,
                        grid_size=grid, scheme=scheme,
                        dt=anim["dt"], frames=anim["frames"],
                        interval=anim["interval"], show_labels=show_labels,
                        background=background, res_factor=anim["res_factor"],
                        orbital_type=self.orbital,
                        real_component=self.real_component,
                    )
                    self._anim = ani

            elif plot_id == 5:
                if mode == "s":
                    fig = plot5.plot5_volume_density(
                        calc, n, l, m, self.scale,
                        resolution=anim["volume_res"], scheme=scheme,
                        background=background, show_labels=show_labels,
                        smooth_sigma=anim["sigma"],
                        surface_count=anim["surface_count"],
                        opacity=anim["opacity"],
                        orbital_type=self.orbital,
                        real_component=self.real_component,
                    )
                else:
                    fig = plot5.plot5_volume_density_dynamic(
                        calc, n, l, m, self.scale,
                        resolution=anim["volume_res"],
                        dt=anim["dt"], frames=anim["volume_frames"],
                        mode=anim["component"], scheme=scheme,
                        background=background, show_labels=show_labels,
                        smooth_sigma=anim["sigma"],
                        surface_count=anim["surface_count"],
                        opacity=anim["opacity"],
                        orbital_type=self.orbital,
                        real_component=self.real_component,
                    )

            elif plot_id == 6:
                if mode == "s":
                    fig = plot6.plot6_volume_shell(
                        calc, n, l, m, self.scale,
                        resolution=anim["volume_res"], scheme=scheme,
                        background=background, show_labels=show_labels,
                        smooth_sigma=anim["sigma"],
                        surface_count=anim["surface_count"],
                        opacity=anim["opacity"],
                        orbital_type=self.orbital,
                        real_component=self.real_component,
                    )
                else:
                    fig = plot6.plot6_volume_shell_dynamic(
                        calc, n, l, m, self.scale,
                        resolution=anim["volume_res"],
                        dt=anim["dt"], frames=anim["volume_frames"],
                        mode=anim["component"], scheme=scheme,
                        background=background, show_labels=show_labels,
                        smooth_sigma=anim["sigma"],
                        surface_count=anim["surface_count"],
                        opacity=anim["opacity"],
                        orbital_type=self.orbital,
                        real_component=self.real_component,
                    )

            elif plot_id == 7:
                scheme_d = resolve_scheme(scheme)
                cmap = scheme_d["shell"]
                if scheme_d["key"] == "classic":
                    cmap = "inferno"
                if scheme_d["key"] == "signed":
                    cmap = "turbo"
                fig = pointcloud_static.render_pointcloud(
                    n=n, l=l, m=m,
                    resolution=anim["volume_res"], cmap=cmap,
                    cut_mode="none", background=background,
                    color_by="shell", orbital_type=self.orbital,
                    real_component=self.real_component,
                    show_labels=show_labels, save=None, show=False,
                    verbose=False,
                )
            else:
                return

            if fig is None:
                messagebox.showerror("Plot error", "The returned figure is empty.")
                self.status_var.set("Plot failed")
                return

            if plot_id in (5, 6):
                fig.show()
            else:
                try:
                    fig.canvas.manager.set_window_title(
                        f"plot{plot_id}  n={n}, l={l}, m={m}, {self.orbital}, {plane}"
                    )
                except Exception:
                    pass
                fig.show()
                self._figs.append(fig)
                if len(self._figs) > 12:
                    self._figs = self._figs[-12:]

            self.status_var.set(
                f"plot{plot_id} ({'static' if mode == 's' else 'dynamic'}) displayed"
            )
        except Exception as exc:
            messagebox.showerror("Plot error", f"Exception while generating the figure:\n{exc}")
            self.status_var.set("Plot failed")
        finally:
            self.root.config(cursor="")
            self.root.update_idletasks()


def main():
    root = tk.Tk()
    HydrogenGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
