"""Reach spectroscopy — device-agnostic machine identification.  [SI S9/S14]

Measures the *machine abstraction* ``M = (D, {phi_n})`` of any reservoir
substrate from a single-tone probe run.  The analysis is **loader-independent**:
it operates on parsed time-series arrays (an actuation ``drive`` and a
multi-channel ``states`` readout), so the identical code path runs on a
simulated lattice, a shuffled control, or digitized experimental data — the
"one pipeline, many substrates" requirement of SI S14.

Quick-start
-----------
>>> from openprc.analysis import reach_spectroscopy
>>> from openprc.reservoir import StateLoader, features as feat
>>>
>>> loader = StateLoader("experiments/example/output/simulation.h5")
>>> states = feat.NodeDisplacements(reference_node=0).transform(loader)  # (T, N)
>>> drive  = loader.get_actuation_signal(actuator_idx=0, dof=1)          # (T,)
>>>
>>> M = reach_spectroscopy(states, t=loader.time, drive=drive)
>>> M.D                     # reach per tolerance, e.g. {0.01: 7, 0.001: 5}
>>> M.summary()             # per-harmonic table
>>> M.plot()                # the six-panel scientific picture
>>> M.save("experiments/example/output")

Masking functionality
----------------------
Every measurement is behind a boolean flag; disable what you don't need for a
faster, leaner run (e.g. skip the O(T·N) capacity fit or the input FFT)::

    M = reach_spectroscopy(states, dt=0.005, drive=drive,
                           capacities=True, quadrature=True,
                           noise_floor=False, rank=False, input_spectrum=False)

Efficiency
----------
The per-harmonic capacity fit shares a single least-squares factorization of
the readout across *all* harmonics and both quadrature channels (one batched
``lstsq`` with ``2·n_max`` right-hand sides instead of ``2·n_max`` separate
solves), and the quadrature geometry uses a closed-form 2x2 eigen-solve
vectorized over harmonics — same numbers as the original per-harmonic
``svd``/``lstsq`` loops, a large constant-factor faster.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

MAROON, ORANGE, GRAY = "#861F41", "#E87722", "#75787B"  # VT brand


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting. Install: pip install matplotlib"
        ) from e


# ═══════════════════════════════════════════════════════════════════════════
#  Fundamental-frequency estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_f0(x, dt, thresh=0.05, kmax=8):
    """Fundamental = approximate GCD of the significant drive tones.

    For a single-tone probe the strongest peak *is* ``f0``; for a multi-tone
    input (e.g. primes ``{2,3,5}·f0``) the fundamental is the GCD of the tones,
    which no single peak reveals — hence the divisor search.
    """
    x = np.asarray(x, float)
    T = len(x)
    X = np.abs(np.fft.rfft(x - x.mean()))
    fr = np.fft.rfftfreq(T, dt)
    peaks = fr[X > thresh * X.max()]
    peaks = peaks[peaks > 0]
    if peaks.size == 0:
        raise ValueError("no spectral peaks found in drive; pass f0 explicitly")
    fmin = peaks.min()
    for k in range(1, kmax + 1):  # try f0 = fmin/k, largest first
        cand = fmin / k
        ratio = peaks / cand
        if np.all(np.abs(ratio - np.round(ratio)) < 0.02):
            return float(cand)
    return float(fmin)


# ═══════════════════════════════════════════════════════════════════════════
#  Core (loader-independent, plain arrays)
# ═══════════════════════════════════════════════════════════════════════════

def spectroscopy_core(x, y, dt, f0=None, settle_frac=0.5,
                      eps=(1e-2, 1e-3), n_max=None,
                      capacities=True, quadrature=True, noise_floor=True,
                      rank=True, input_spectrum=True):
    """Measure the machine abstraction from parsed arrays.

    Parameters
    ----------
    x : (T,) array
        Actuation / drive reference (used for f0 estimation and the input
        spectrum panel).
    y : (T, N) array
        Multi-channel reservoir state (the readout).
    dt : float
        Uniform sample interval [s].
    f0 : float, optional
        Fundamental drive frequency [Hz].  Estimated from ``x`` if omitted.
    settle_frac : float
        Fraction of the record discarded as transient before analysis.
    eps : sequence of float
        Reach tolerances; ``D(eps)`` is the largest harmonic whose held-out
        capacity exceeds ``1 - eps`` in *both* quadrature channels.
    n_max : int, optional
        Highest harmonic probed.  Defaults to just below Nyquist (cap 40).
    capacities, quadrature, noise_floor, rank, input_spectrum : bool
        Masking flags — skip a measurement to run faster / leaner.  ``D`` and
        the reconstruction panels require ``capacities``.

    Returns
    -------
    dict
        Raw measurement dict (wrapped by :class:`MachineAbstraction`).
    """
    eps = tuple(eps)
    T = min(len(x), y.shape[0])
    x = np.asarray(x[:T], float)
    y = np.asarray(y[:T], float)
    t = np.arange(T) * dt

    # -- fundamental: given or estimated from the drive spectrum
    if f0 is None:
        f0 = estimate_f0(x, dt)
    w0 = 2 * np.pi * f0

    # -- steady window: an integer number of fundamental periods, split in half.
    #    A multiple-of-4 period count keeps both halves an even number of
    #    periods, so the half-integer noise probes stay orthogonal to signal.
    i0 = int(settle_frac * T)
    n_per = int((t[-1] - t[i0]) * f0)
    if n_per < 8:
        raise ValueError("window too short: need >= 8 fundamental periods "
                         f"after settle (got {n_per})")
    n_per -= n_per % 4
    i1 = i0 + int(round(n_per / f0 / dt))
    tw, yw, xw = t[i0:i1], y[i0:i1], x[i0:i1]
    yw = yw - yw.mean(axis=0)
    imid = len(tw) // 2
    t_tr, y_tr = tw[:imid], yw[:imid]
    t_te, y_te = tw[imid:], yw[imid:]

    if n_max is None:  # probe past the visible ladder, below Nyquist
        nyq = 0.5 / dt
        n_max = int(min(nyq / f0 - 1, 40))
    n_max = max(int(n_max), 1)
    n_arr = np.arange(1, n_max + 1)

    out = dict(f0=float(f0), dt=float(dt), eps=list(eps), n_max=n_max,
               N=int(y.shape[1]), w0=w0,
               t_tr=t_tr, y_tr=y_tr, t_te=t_te, y_te=y_te)

    # -- per-state complex amplitudes at each harmonic (scored on test half).
    #    A[k] is the phasor of every state channel at harmonic (k+1).
    E = np.exp(-1j * np.outer(n_arr * w0, t_te))       # (n_max, Tte)
    A = 2.0 * (E @ y_te) / len(t_te)                   # (n_max, N) complex
    amp_max = np.abs(A).max(axis=1)
    amp_rms = np.sqrt((np.abs(A) ** 2).mean(axis=1))
    out.update(amp_max=amp_max, amp_rms=amp_rms)

    # dominant-state phase per harmonic (the {phi_n} of the abstraction)
    dom = np.argmax(np.abs(A), axis=1)
    out["phi_n"] = np.angle(A[np.arange(n_max), dom])

    # -- noise floor: same projection at half-integer harmonics (no signal)
    if noise_floor:
        Eh = np.exp(-1j * np.outer((n_arr + 0.5) * w0, t_te))
        noise = np.median(np.abs(2.0 * (Eh @ y_te) / len(t_te)), axis=1)
        out["noise"] = noise
        out["sig"] = amp_max > 3 * noise
    else:
        out["noise"] = None
        out["sig"] = np.ones(n_max, dtype=bool)

    # -- quadrature geometry: for each harmonic, B = [Re A; Im A] (2 x N).
    #    s2/s1 measures the anisotropy of the spanned 2-space; the singular
    #    values are sqrt-eigenvalues of the 2x2 Gram matrix [[a,b],[b,c]],
    #    solved in closed form and vectorized over harmonics. [SI S8]
    if quadrature:
        Re, Im = A.real, A.imag
        a = np.einsum("kn,kn->k", Re, Re)
        b = np.einsum("kn,kn->k", Re, Im)
        c = np.einsum("kn,kn->k", Im, Im)
        half_tr = 0.5 * (a + c)
        disc = np.sqrt(np.maximum(half_tr ** 2 - (a * c - b * b), 0.0))
        s1 = np.sqrt(np.maximum(half_tr + disc, 0.0))
        s2 = np.sqrt(np.maximum(half_tr - disc, 0.0))
        quad = np.where(s1 > 0, s2 / s1, 0.0)
        out["quad"] = quad
        out["phi_eff"] = 2 * np.arctan(quad)
    else:
        out["quad"] = None
        out["phi_eff"] = None

    # -- two-channel capacities: fit on the train half, SCORE on the held-out
    #    test half.  All harmonics x both quadrature channels share ONE
    #    least-squares factorization of the readout: a single batched lstsq
    #    with 2*n_max right-hand sides.
    if capacities:
        NW_tr = w0 * np.outer(t_tr, n_arr)             # (Ttr, n_max)
        NW_te = w0 * np.outer(t_te, n_arr)
        # channel 1: sin(n w0 t);  channel 2: sin(n w0 t - pi/2) = -cos
        G_tr = np.hstack([np.sin(NW_tr), np.sin(NW_tr - np.pi / 2)])  # (Ttr, 2n)
        G_te = np.hstack([np.sin(NW_te), np.sin(NW_te - np.pi / 2)])
        Xtr = np.hstack([y_tr, np.ones((len(t_tr), 1))])
        Xte = np.hstack([y_te, np.ones((len(t_te), 1))])
        W, *_ = np.linalg.lstsq(Xtr, G_tr, rcond=None)  # (N+1, 2n)
        c_tr = 1 - ((G_tr - Xtr @ W) ** 2).sum(0) / (G_tr ** 2).sum(0)
        c_te = 1 - ((G_te - Xte @ W) ** 2).sum(0) / (G_te ** 2).sum(0)
        # column layout matches the original: (c1_train, c1_test, c2_train, c2_test)
        C = np.column_stack([c_tr[:n_max], c_te[:n_max],
                             c_tr[n_max:], c_te[n_max:]])
        out["C"] = C

        D = {}
        for e in eps:
            passing = np.minimum(C[:, 1], C[:, 3]) > 1 - e
            D[e] = int(np.max(np.nonzero(passing)[0]) + 1) if passing.any() else 0
        out["D"] = D
    else:
        out["C"] = None
        out["D"] = {}

    # -- input spectrum (probe purity / prime-tone comb)
    if input_spectrum:
        Xin = np.abs(np.fft.rfft(xw - xw.mean())) / len(xw)
        fin = np.fft.rfftfreq(len(xw), dt)
        out["input_spec"] = (fin, Xin)
    else:
        out["input_spec"] = None

    # -- state effective rank: the N of the sum rule [SI Eq. sumrule]
    if rank:
        sv = np.linalg.svd(y_te / np.sqrt(len(t_te)), compute_uv=False)
        p = sv ** 2 / np.sum(sv ** 2)
        n_eff = float(np.exp(-np.sum(p * np.log(p + 1e-300))))  # participation
        out["sv"] = sv
        out["n_eff"] = n_eff
    else:
        out["sv"] = None
        out["n_eff"] = None

    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MachineAbstraction:
    """The measured machine abstraction ``M = (D, {phi_n})`` plus diagnostics.

    Wraps the raw measurement dict from :func:`spectroscopy_core` with
    convenient attribute access, a summary table, plotting, and export.
    Attributes not computed (masked by a flag) are ``None``.
    """

    data: dict
    x_is_drive: bool = False
    channel_names: Optional[Sequence[str]] = None
    title: Optional[str] = None

    # ── attribute access ────────────────────────────────────────────────
    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    @property
    def f0(self) -> float:
        return self.data["f0"]

    @property
    def D(self) -> dict:
        return self.data["D"]

    @property
    def phi_n(self) -> np.ndarray:
        return self.data["phi_n"]

    @property
    def C(self):
        return self.data["C"]

    @property
    def n_max(self) -> int:
        return self.data["n_max"]

    @property
    def N(self) -> int:
        return self.data["N"]

    @property
    def n_eff(self):
        return self.data["n_eff"]

    # ── summary ─────────────────────────────────────────────────────────
    def summary(self, printout=True) -> str:
        d = self.data
        lines = [f"f0 = {d['f0']:.4f} Hz | N = {d['N']} states"
                 + (f" | N_eff = {d['n_eff']:.1f}" if d['n_eff'] is not None else "")]
        header = " n |"
        if d["C"] is not None:
            header += "  C_n(test) | C_n^phi(test) |"
        if d["quad"] is not None:
            header += " quad s2/s1 |"
        header += "  amp_max"
        lines.append(header)
        for k in range(d["n_max"]):
            row = f"{k + 1:2d} |"
            if d["C"] is not None:
                row += f"  {d['C'][k, 1]:9.6f} | {d['C'][k, 3]:12.6f} |"
            if d["quad"] is not None:
                row += f" {d['quad'][k]:9.3f}  |"
            row += f" {d['amp_max'][k]:.3e}"
            lines.append(row)
        for e, dd in d["D"].items():
            lines.append(f"D(eps={e:g}) = {dd}")
        text = "\n".join(lines)
        if printout:
            print(text)
        return text

    def __repr__(self) -> str:
        d = self.data
        reach = ", ".join(f"{e:g}:{v}" for e, v in d["D"].items()) or "n/a"
        return (f"MachineAbstraction(f0={d['f0']:.4g} Hz, N={d['N']}, "
                f"n_max={d['n_max']}, D=[{reach}])")

    # ── export ──────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """JSON-safe frozen abstraction (the reproducible fingerprint)."""
        d = self.data
        keep = max(d["D"].values() or [0])
        frozen = dict(
            f0=d["f0"], eps=d["eps"], N=d["N"], n_eff=d["n_eff"],
            D={str(k): v for k, v in d["D"].items()},
            phi_n=d["phi_n"][:keep].tolist(),
        )
        if d["quad"] is not None:
            frozen["quad"] = d["quad"][:keep].tolist()
        return frozen

    def save(self, dest, prefix="reach_spectroscopy"):
        """Write ``<prefix>.npz`` (arrays) and ``machine_abstraction.json``.

        ``dest`` may be a directory (files are placed inside) or a full path
        stem.  Returns the directory the files landed in.
        """
        dest = Path(dest)
        out_dir = dest if dest.is_dir() or dest.suffix == "" else dest.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in self.data.items()
                  if isinstance(v, (np.ndarray, float, int))}
        np.savez(out_dir / f"{prefix}.npz", **arrays)
        (out_dir / "machine_abstraction.json").write_text(
            json.dumps(self.to_dict(), indent=2))
        return out_dir

    # ── plotting ────────────────────────────────────────────────────────
    def plot(self, panels=None, save_to=None):
        """Six-panel scientific picture.  ``panels`` selects a subset by key:
        ``{"reach","capacities","quadrature","spectrum","reconstruction",
        "rank"}``.  Panels whose data was masked are skipped automatically."""
        return plot_spectroscopy(self.data, title=self.title or "Reach spectroscopy",
                                 x_is_drive=self.x_is_drive, panels=panels,
                                 save_to=save_to)

    def plot_harmonic_fits(self, n_show=None, periods=2, save_to=None):
        """Per-harmonic train-vs-held-out reconstruction stack."""
        return plot_harmonic_fits(self.data, n_show=n_show, periods=periods,
                                  save_to=save_to)


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting — the full scientific picture
# ═══════════════════════════════════════════════════════════════════════════

_ALL_PANELS = ("reach", "capacities", "quadrature",
               "spectrum", "reconstruction", "rank")


def plot_spectroscopy(M, title="Reach spectroscopy", x_is_drive=False,
                      panels=None, save_to=None):
    plt = _require_matplotlib()
    panels = set(panels) if panels is not None else set(_ALL_PANELS)
    n = np.arange(1, M["n_max"] + 1)
    C, D = M.get("C"), M.get("D", {})
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    def _off(a, msg):
        a.text(0.5, 0.5, msg, ha="center", va="center", color=GRAY,
               fontsize=10, transform=a.transAxes)
        a.set_xticks([]); a.set_yticks([])

    # A — reach spectrum: state harmonic content vs noise floor
    a = ax[0, 0]
    if "reach" in panels:
        a.semilogy(n, M["amp_max"], "o-", color=MAROON, ms=4, label="max over states")
        a.semilogy(n, M["amp_rms"], "s--", color=ORANGE, ms=3, label="rms over states")
        if M.get("noise") is not None:
            a.semilogy(n, M["noise"], ":", color=GRAY, label="noise floor (n+1/2)")
        for e, d in D.items():
            if d:
                a.axvline(d + 0.5, color=MAROON, lw=0.8, alpha=0.5)
        a.set(xlabel="harmonic $n$", ylabel="state amplitude", title="A  Reach spectrum")
        a.legend(fontsize=8)
    else:
        _off(a, "A  (masked)")

    # B — two-channel capacities, train vs held-out test
    b = ax[0, 1]
    if "capacities" in panels and C is not None:
        b.plot(n, C[:, 1], "o-", color=MAROON, ms=4, label="$C_n$ (test)")
        b.plot(n, C[:, 3], "s-", color=ORANGE, ms=4, label=r"$C_n^{\phi}$ (test)")
        b.plot(n, C[:, 0], "-", color=MAROON, alpha=0.25, label="train")
        b.plot(n, C[:, 2], "-", color=ORANGE, alpha=0.25)
        for e, d in D.items():
            b.axhline(1 - e, color=GRAY, lw=0.8, ls=":")
            if d:
                b.annotate(rf"$\mathcal{{D}}(\epsilon={e:g}) = {d}$", (d, 1 - e),
                           textcoords="offset points", xytext=(6, -12), fontsize=9)
        b.set(xlabel="harmonic $n$", ylabel="capacity", ylim=(-0.05, 1.05),
              title="B  Per-harmonic capacities (held-out)")
        b.legend(fontsize=8)
    else:
        _off(b, "B  capacities (masked)")

    # C — quadrature quality & compiler conditioning [SI S8]
    c = ax[0, 2]
    if "quadrature" in panels and M.get("quad") is not None:
        sig = M["sig"]
        q = np.where(sig, M["quad"], np.nan)
        c.plot(n, q, "o-", color=MAROON, ms=4)
        c.plot(n[~sig], M["quad"][~sig], "x", color=GRAY, ms=5,
               label="below noise floor (isotropic)")
        c.legend(fontsize=7)
        c.set(xlabel="harmonic $n$",
              ylabel=r"$\sigma_2/\sigma_1$  ($=\tan(\Delta\psi_{\rm eff}/2)$)",
              title="C  Quadrature quality / conditioning", ylim=(0, 1.05))
        c2 = c.twinx()
        with np.errstate(divide="ignore"):
            c2.semilogy(n, np.where(sig & (M["quad"] > 0), M["quad"] ** -2, np.nan),
                        "s--", color=ORANGE, ms=3)
        c2.set_ylabel(r"$\kappa = (\sigma_1/\sigma_2)^2$", color=ORANGE)
    else:
        _off(c, "C  quadrature (masked)")

    # D — input spectrum (probe purity / prime tones)
    d = ax[1, 0]
    if "spectrum" in panels and M.get("input_spec") is not None:
        fin, Xin = M["input_spec"]
        d.semilogy(fin / M["f0"], Xin + 1e-300, color=GRAY, lw=0.8)
        lbl = "Input spectrum (driven node)" if x_is_drive \
            else "$f_0$ reference channel spectrum (strongest state)"
        d.set(xlabel="frequency / $f_0$", ylabel="amplitude",
              xlim=(0, M["n_max"] + 1), title="D  " + lbl)
    else:
        _off(d, "D  input spectrum (masked)")

    # E — reconstruction examples: last passing vs first failing harmonic
    e_ax = ax[1, 1]
    if "reconstruction" in panels and C is not None and D:
        d0 = D[M["eps"][0]]
        t_te, y_te, w0 = M["t_te"], M["y_te"], M["w0"]
        Xte = np.hstack([y_te, np.ones((len(t_te), 1))])
        for nn, col, lab in [(max(d0, 1), MAROON, "last passing"),
                             (min(d0 + 1, M["n_max"]), ORANGE, "first failing")]:
            g = np.sin(nn * w0 * t_te)
            w, *_ = np.linalg.lstsq(Xte, g, rcond=None)
            sl = slice(0, int(2 / M["f0"] / M["dt"]))  # two periods
            e_ax.plot(t_te[sl], g[sl], color=GRAY, lw=0.8)
            e_ax.plot(t_te[sl], (Xte @ w)[sl], color=col, lw=1.2,
                      label=f"n={nn} ({lab})")
        e_ax.set(xlabel="t (s)", ylabel="target vs reconstruction",
                 title="E  Reconstructions at the reach boundary")
        e_ax.legend(fontsize=8)
    else:
        _off(e_ax, "E  reconstruction (masked)")

    # F — state singular spectrum: the N of the sum rule [SI Eq. sumrule]
    f = ax[1, 2]
    if "rank" in panels and M.get("sv") is not None:
        f.semilogy(np.arange(1, len(M["sv"]) + 1), M["sv"], ".-", color=MAROON, ms=3)
        f.axvline(M["n_eff"], color=ORANGE, lw=1)
        f.annotate(f"$N_{{\\rm eff}}$ = {M['n_eff']:.1f}  (of N = {M['N']})",
                   (M["n_eff"], M["sv"][0]), textcoords="offset points",
                   xytext=(6, -4), fontsize=9, color=ORANGE)
        f.set(xlabel="index", ylabel="singular value", title="F  State effective rank")
    else:
        _off(f, "F  effective rank (masked)")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=160)
        print(f"figure -> {save_to}")
    return fig


def plot_harmonic_fits(M, n_show=None, periods=2, save_to=None):
    """One row per harmonic: target vs reconstruction on the TRAIN segment
    (left, fit here) and the held-out TEST segment (right, scored here).
    Overfitting is visible as train tracking while test drifts."""
    plt = _require_matplotlib()
    D0 = max(M.get("D", {}).values() or [0])
    n_show = n_show or min(D0 + 2, 12)
    n_show = min(n_show, M["n_max"])
    w0, dt, f0 = M["w0"], M["dt"], M["f0"]
    t_tr, y_tr, t_te, y_te = M["t_tr"], M["y_tr"], M["t_te"], M["y_te"]
    Xtr = np.hstack([y_tr, np.ones((len(t_tr), 1))])
    Xte = np.hstack([y_te, np.ones((len(t_te), 1))])
    sl = slice(0, int(round(periods / f0 / dt)))

    fig, ax = plt.subplots(n_show, 2, figsize=(11, 1.25 * n_show),
                           sharex="col", squeeze=False)
    for i, n in enumerate(range(1, n_show + 1)):
        g_tr, g_te = np.sin(n * w0 * t_tr), np.sin(n * w0 * t_te)
        w, *_ = np.linalg.lstsq(Xtr, g_tr, rcond=None)
        for j, (tt, gg, XX, col, lab) in enumerate(
                [(t_tr, g_tr, Xtr, MAROON, "train (fit)"),
                 (t_te, g_te, Xte, ORANGE, "test (held out)")]):
            a = ax[i, j]
            a.plot(tt[sl], gg[sl], color=GRAY, lw=0.9)
            a.plot(tt[sl], (XX @ w)[sl], color=col, lw=1.1)
            cc = 1 - np.sum((gg - XX @ w) ** 2) / np.sum(gg ** 2)
            a.annotate(f"$C_{{{n}}}$={cc:.4f}", (0.02, 0.72),
                       xycoords="axes fraction", fontsize=8)
            a.set_yticks([])
            if i == 0:
                a.set_title(lab, fontsize=10)
        ax[i, 0].set_ylabel(f"n={n}", rotation=0, labelpad=16, fontsize=9)
    ax[-1, 0].set_xlabel("t (s)")
    ax[-1, 1].set_xlabel("t (s)")
    fig.suptitle("Per-harmonic reconstructions — train vs held-out test")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=160)
        print(f"figure -> {save_to}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  Public entry point — takes PARSED time-series, not an experiment path
# ═══════════════════════════════════════════════════════════════════════════

def reach_spectroscopy(states, dt=None, *, t=None, drive=None, f0=None,
                       settle_frac=0.5, eps=(1e-2, 1e-3), n_max=None,
                       capacities=True, quadrature=True, noise_floor=True,
                       rank=True, input_spectrum=True,
                       prune_dead=True, prune_tol=1e-10,
                       channel_names=None, title=None,
                       verbose=True) -> MachineAbstraction:
    """Run reach spectroscopy on parsed reservoir time-series.

    Parameters
    ----------
    states : array (T, N) or (T, N, 3)
        The multi-channel reservoir readout.  A 3-D position array is flattened
        to ``(T, 3N)`` automatically.
    dt : float, optional
        Uniform sample interval [s].  Provide this *or* ``t``.
    t : (T,) array, optional
        Sample times; ``dt`` is taken as ``median(diff(t))``.
    drive : (T,) array, optional
        Actuation reference — its spectrum drives ``f0`` estimation and the
        input-spectrum panel.  If ``None``, the strongest state channel is used
        as the ``f0`` reference (spectroscopy is unaffected; the GCD estimator
        reads ``f0`` from the harmonic comb either way).
    f0 : float, optional
        Fundamental drive frequency [Hz]; estimated if omitted.
    capacities, quadrature, noise_floor, rank, input_spectrum : bool
        Masking flags for individual measurements (see :func:`spectroscopy_core`).
    prune_dead : bool
        Drop quiescent state channels (fixed nodes / dead DOFs) before fitting —
        a large least-squares speed-up with no effect on the result.
    verbose : bool
        Print a per-harmonic summary table.

    Returns
    -------
    MachineAbstraction
    """
    states = np.asarray(states)
    if states.ndim == 3:               # (T, N, 3) -> (T, 3N)
        states = states.reshape(states.shape[0], -1)
    elif states.ndim != 2:
        raise ValueError(f"states must be (T, N) or (T, N, 3); got {states.shape}")
    y = np.asarray(states, float)

    if dt is None:
        if t is None:
            raise ValueError("provide either dt or t")
        t = np.asarray(t, float).ravel()
        dt = float(np.median(np.diff(t)))

    # drive reference: explicit actuation, or the strongest state channel
    if drive is not None:
        x = np.asarray(drive, float).ravel()
        x_is_drive = True
    else:
        x = y[:, int(np.argmax(y.std(axis=0)))].copy()
        x_is_drive = False

    # prune quiescent channels: fixed nodes / dead DOFs contribute nothing but
    # inflate the least-squares problem
    names = list(channel_names) if channel_names is not None else None
    if prune_dead:
        std = y.std(axis=0)
        keep = std > prune_tol * std.max() if std.max() > 0 else np.ones(y.shape[1], bool)
        if verbose and keep.sum() < y.shape[1]:
            print(f"states: {keep.sum()} active of {y.shape[1]} channels "
                  f"(dt = {dt:.5f} s)")
        y = y[:, keep]
        if names is not None:
            names = [nm for nm, k in zip(names, keep) if k]

    data = spectroscopy_core(x, y, dt=dt, f0=f0, settle_frac=settle_frac,
                             eps=eps, n_max=n_max, capacities=capacities,
                             quadrature=quadrature, noise_floor=noise_floor,
                             rank=rank, input_spectrum=input_spectrum)

    M = MachineAbstraction(data=data, x_is_drive=x_is_drive,
                           channel_names=names, title=title)
    if verbose:
        M.summary()
    return M