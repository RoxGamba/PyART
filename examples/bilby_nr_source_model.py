"""
Validate the PyART bilby source model

The GRAthena++ catalogue is arXiv:2411.11989: q = 1, 2, 3, 4, non-spinning,
quasi-circular. The SXS partners are matched by mass ratio.

Needs network access on the first run, to download the simulations.

    python examples/bilby_nr_source_model.py --gra_id 0001
"""

import argparse
import os
import subprocess

import numpy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

import lal

from PyART.logging_config import setup_logging
from PyART.utils.wf_utils import align_phase
from PyART.plugin.bilby_plugin import (
    component_masses_and_spins,
    compute_conditioning_parameters,
    estimate_time_of_maximum_amplitude,
    load_nr_waveform,
    nr_frequency_domain_source_model,
    nr_time_domain_polarisations,
)

setup_logging(level="INFO")

# GRAthena++ ID -> the SXS simulation at the same mass ratio, all non-spinning.
SXS_PARTNER = {
    "0001": "1155",  # q = 1, 40.65 orbits, ecc 6.9e-06
    "0002": "1167",  # q = 2, 40.52 orbits, ecc 3.6e-04
    "0003": "1179",  # q = 3, 15.68 orbits, ecc 3.7e-05
    "0004": "0167",  # q = 4, 15.59 orbits, ecc 9.5e-05
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gra-id", default="0001", help="GRAthena++ ID")
parser.add_argument(
    "--sxs-id", default=None, help="SXS ID (default: the partner of --gra-id)"
)
parser.add_argument("--res", default="128", help="GRAthena++ resolution")
parser.add_argument("--ext", default="CCE", help="GRAthena++ extraction: ext, CCE or finite")
parser.add_argument("--r-ext", default="50.00", help="GRAthena++ extraction radius")
parser.add_argument(
    "--total_mass",
    type=float,
    default=300.0,
    help="Total mass (Msun). The default puts the whole GRAthena++ record inside "
    "[f_min, f_max] while still leaving room for the conditioning taper: the "
    "record starts at 20.4 Hz and the Stage-1 taper eats 0.25 s of 0.63 s. "
    "Higher masses push the start of the record below f_min, lower ones leave "
    "nothing after the taper.",
)
parser.add_argument("--distance", type=float, default=400.0, help="Distance (Mpc)")
parser.add_argument(
    "--iota",
    type=float,
    default=1.2,
    help="Inclination (rad). Face-on by default: h+ - i hx is then a clean "
    "circularly polarised chirp, so A and phi are unambiguous.",
)
parser.add_argument("--phase", type=float, default=0.0, help="Orbital phase (rad)")
parser.add_argument("--ellmax", type=int, default=4, help="Maximum ell to load")
parser.add_argument("--cut_U", type=float, default=200.0, help="Junk-radiation cut (M)")
parser.add_argument(
    "--t_after_peak", type=float, default=100.0, help="Keep this much after merger (M)"
)
parser.add_argument(
    "--settle_cycles",
    type=float,
    default=0.5,
    help="Extra cycles at f_min to skip after the Stage-1 taper, before the "
    "comparison window starts",
)
parser.add_argument(
    "--align_end_before",
    type=float,
    default=100.0,
    help="End the alignment window this far before the merger (M)",
)
parser.add_argument("--duration", type=float, default=8.0, help="Segment duration (s)")
parser.add_argument("--srate", type=float, default=4096.0, help="Sampling rate (Hz)")
parser.add_argument(
    "--f-min",
    type=float,
    default=15.0,
    help="Minimum frequency (Hz). Must sit *below* the frequency at which the NR "
    "record starts (20.4 Hz for GRAthena:BHBH:0001 at the default mass), or the "
    "conditioning high-pass acts on the whole waveform rather than on the empty "
    "band beneath it. --f_min_scan shows what that costs.",
)
parser.add_argument(
    "--f-min-scan",
    type=float,
    nargs="*",
    default=[15.0, 17.0, 20.0],
    help="Extra f_min values to repeat the round-trip test at",
)
parser.add_argument(
    "--f_max", type=float, default=1024.0, help="Maximum frequency (Hz)"
)
parser.add_argument("--no_plot", action="store_true", help="Skip the figures")
args = parser.parse_args()

sxs_id = args.sxs_id or SXS_PARTNER[args.gra_id]

repo_path = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
out_path = os.path.join(repo_path, "examples/local_data/")

frequency_array = numpy.arange(
    0.0, args.srate / 2 + args.duration**-1, 1.0 / args.duration
)
band = (frequency_array >= args.f_min) & (frequency_array <= args.f_max)
geometric_to_seconds = args.total_mass * lal.MTSUN_SI
segment_time = (
    numpy.arange(int(round(args.duration * args.srate))) / args.srate - args.duration
)


gra_path = os.path.join(repo_path, "examples/local_data/gra/")
sxs_path = os.path.join(repo_path, "examples/local_data/sxs/")

# Waveform_GRA re-fetches the whole resolution tarball every time it is
# constructed with download=True -- it has no "already on disk" check, unlike
# Waveform_SXS, and the archive is a few hundred MB. So decide from the
# filesystem instead, and only ask for the download when the strain file that
# will actually be read is missing.
_gra_strain = {
    "ext": "rh_Asymptotic_GeometricUnits.h5",
    "CCE": "rh_CCE_GeometricUnits.h5",
    "finite": "rh_FiniteRadii_GeometricUnits.h5",
}[args.ext]
downloaded = {
    "gra": os.path.exists(
        os.path.join(gra_path, f"GRA_BHBH_{args.gra_id}", args.res, _gra_strain)
    ),
    "sxs": False,
}


def gra_arguments(**overrides):
    arguments = dict(
        catalog="gra",
        ID=args.gra_id,
        path=gra_path,
        download=not downloaded["gra"],
        res=args.res,
        ext=args.ext,
        r_ext=args.r_ext,
        ellmax=args.ellmax,
        cut_U=args.cut_U,
        t_after_peak=args.t_after_peak,
        minimum_frequency=args.f_min,
        maximum_frequency=args.f_max,
        sampling_frequency=args.srate,
    )
    arguments.update(overrides)
    return arguments


def sxs_arguments(**overrides):
    arguments = dict(
        catalog="sxs",
        ID=sxs_id,
        path=sxs_path,
        download=not downloaded["sxs"],
        ellmax=args.ellmax,
        load_m0=False,
        cut_U=args.cut_U,
        t_after_peak=args.t_after_peak,
        minimum_frequency=args.f_min,
        maximum_frequency=args.f_max,
        sampling_frequency=args.srate,
    )
    arguments.update(overrides)
    return arguments


# ─────────────────────────────────────────────────────────────────────────────
# Amplitude, phase, and the alignment they are computed after
# ─────────────────────────────────────────────────────────────────────────────


def amplitude_and_phase(hplus, hcross):
    """
    A and phi of h = h+ - i hx, in PyART's convention (h = A exp(-i phi)).

    Face-on this is exact: h is a circularly polarised chirp and A, phi are the
    amplitude and phase of the (2, 2) mode. Off-axis the higher modes beat
    against each other and A picks up a modulation -- still well defined, just
    less clean to read.
    """
    signal = hplus - 1j * hcross
    return numpy.abs(signal), -numpy.unwrap(numpy.angle(signal))


def align(window, time_a, phase_a, time_b, phase_b, tau_max):
    """
    Align a onto b by minimising

        chi2(tau) = int_window [ phi_a(t + tau) - phi_b(t) - dphi ]^2 dt

    over the time shift tau and the phase offset dphi. dphi is solved
    analytically at each tau by PyART.utils.wf_utils.align_phase; tau comes from
    a bounded scalar minimisation, so it is sub-sample rather than limited to the
    grid spacing (which is what PyART's own wf_utils.Align loop gives).

    Returns (tau, dphi): evaluate a at t + tau and subtract dphi from its phase.
    """
    # align_phase weights on (t >= 0) & (t < Tf), so hand it a shifted window
    local = window - window[0]
    final = local[-1] + (local[1] - local[0])
    resampled_b = numpy.interp(window, time_b, phase_b)

    def chi2(tau):
        shifted_a = numpy.interp(window, time_a + tau, phase_a)
        offset = align_phase(local, final, shifted_a, resampled_b)
        return float(numpy.sum((shifted_a - resampled_b - offset) ** 2))

    solution = minimize_scalar(
        chi2, bounds=(-tau_max, tau_max), method="bounded", options={"xatol": 1e-12}
    )
    tau = float(solution.x)
    shifted_a = numpy.interp(window, time_a + tau, phase_a)
    return tau, float(align_phase(local, final, shifted_a, resampled_b))


def difference(
    window, time_a, amplitude_a, phase_a, time_b, amplitude_b, phase_b, tau_max
):
    """
    Align a onto b, then difference them on `window`.

    Returns the alignment plus dA/A, dphi and |dh|/|h| as arrays over `window`.
    """
    tau, dphi = align(window, time_a, phase_a, time_b, phase_b, tau_max)

    resampled_amplitude_a = numpy.interp(window, time_a + tau, amplitude_a)
    resampled_phase_a = numpy.interp(window, time_a + tau, phase_a) - dphi
    resampled_amplitude_b = numpy.interp(window, time_b, amplitude_b)
    resampled_phase_b = numpy.interp(window, time_b, phase_b)

    signal_a = resampled_amplitude_a * numpy.exp(-1j * resampled_phase_a)
    signal_b = resampled_amplitude_b * numpy.exp(-1j * resampled_phase_b)

    return dict(
        tau=tau,
        dphi=dphi,
        window=window,
        relative_amplitude=(resampled_amplitude_a - resampled_amplitude_b)
        / resampled_amplitude_b,
        phase_difference=resampled_phase_a - resampled_phase_b,
        relative_strain=numpy.abs(signal_a - signal_b) / numpy.abs(signal_b),
    )


def difference_of_polarisations(
    window, time_a, hplus_a, hcross_a, time_b, hplus_b, hcross_b, tau_max
):
    amplitude_a, phase_a = amplitude_and_phase(hplus_a, hcross_a)
    amplitude_b, phase_b = amplitude_and_phase(hplus_b, hcross_b)
    return difference(
        window, time_a, amplitude_a, phase_a, time_b, amplitude_b, phase_b, tau_max
    )


def report(label, result, unit=""):
    print(
        f"  {label:30s} tau = {result['tau']:+.6e}{unit}   dphi = {result['dphi']:+.4f} rad\n"
        f"  {'':30s} |dA/A|   max {numpy.max(numpy.abs(result['relative_amplitude'])):.3e}"
        f"   rms {numpy.std(result['relative_amplitude']):.3e}\n"
        f"  {'':30s} |dphi|   max {numpy.max(numpy.abs(result['phase_difference'])):.3e}"
        f"   rms {numpy.std(result['phase_difference']):.3e}   rad\n"
        f"  {'':30s} |dh|/|h| max {numpy.max(result['relative_strain']):.3e}"
        f"   rms {numpy.std(result['relative_strain']):.3e}"
    )


def to_time_domain(strain):
    """
    Back to the time domain, on `segment_time`. The source model already zeroes
    everything outside [f_min, f_max], so nothing more is masked here.
    """
    return numpy.fft.irfft(strain) * args.srate


def instantaneous_frequency(time, hplus, hcross):
    """GW frequency in Hz, from the derivative of the polarisation phase."""
    _, phase = amplitude_and_phase(hplus, hcross)
    return numpy.abs(numpy.gradient(phase, time)) / (2 * numpy.pi)


# ─────────────────────────────────────────────────────────────────────────────
# Load both simulations and cut them to a common span before merger
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nGRAthena:BHBH:{args.gra_id}  ({args.ext}, r = {args.r_ext}, res {args.res})")
print(f"SXS:BBH:{sxs_id}")

gra = load_nr_waveform(**gra_arguments())
downloaded["gra"] = True
sxs = load_nr_waveform(**sxs_arguments())
downloaded["sxs"] = True

print(
    f"  q: GRA {gra.metadata['q']:.4f}, SXS {sxs.metadata['q']:.4f};  "
    f"chi1z: GRA {gra.metadata['chi1z']:+.2e}, SXS {sxs.metadata['chi1z']:+.2e}"
)
if abs(gra.metadata["q"] - sxs.metadata["q"]) > 1e-3:
    print("  WARNING: mass ratios differ -- the pairing table may be wrong.")


def record_geometry(waveform):
    """(u_start, u_peak, f_gw22 at the start) in geometric units."""
    modes = sorted(waveform.hlm.keys())
    peak = estimate_time_of_maximum_amplitude(waveform.u, waveform.hlm, modes)
    n_edge = min(64, len(waveform.u) // 4)
    frequency = numpy.abs(
        numpy.median(
            numpy.gradient(waveform.hlm[(2, 2)]["p"][:n_edge], waveform.u[:n_edge])
        )
    ) / (2 * numpy.pi)
    return waveform.u[0], peak, frequency


def describe(label, waveform):
    start, peak, frequency = record_geometry(waveform)
    print(
        f"  {label:18s} {start:10.1f} {peak:10.1f} {peak - start:9.1f} "
        f"{frequency:11.4e} {frequency / geometric_to_seconds:9.2f}"
    )
    return start, peak, frequency


print(
    f"\n{'':20s} {'u_start':>10s} {'u_peak':>10s} {'span':>9s} "
    f"{'f_gw22 [M]':>11s} {'f [Hz]':>9s}"
)
gra_start, gra_peak, gra_frequency = describe("GRA before cut", gra)
sxs_start, sxs_peak, sxs_frequency = describe("SXS before cut", sxs)

# Take the shorter of the two spans before merger and cut the longer record to
# match. Merger is the only time origin the two codes agree on.
common_span = min(gra_peak - gra_start, sxs_peak - sxs_start)
gra_kwargs = gra_arguments(cut_U=gra_peak - common_span)
sxs_kwargs = sxs_arguments(cut_U=sxs_peak - common_span)
gra = load_nr_waveform(**gra_kwargs)
sxs = load_nr_waveform(**sxs_kwargs)
gra_start, gra_peak, gra_frequency = describe("GRA after cut", gra)
sxs_start, sxs_peak, sxs_frequency = describe("SXS after cut", sxs)

print(
    f"\n  common span before merger: {common_span:.1f} M = "
    f"{common_span * geometric_to_seconds:.3f} s at M = {args.total_mass:.0f} Msun"
)
if abs(gra_frequency / sxs_frequency - 1) > 0.05:
    print(
        "  WARNING: the two records still start more than 5% apart in frequency; "
        "the cut has not made them comparable."
    )
if min(gra_frequency, sxs_frequency) / geometric_to_seconds < args.f_min:
    print(
        f"  WARNING: the records start below f_min = {args.f_min:.0f} Hz, so the "
        f"conditioning high-pass removes real signal from the beginning. Lower "
        f"--total_mass so that the whole record sits inside the band."
    )

record_start = -common_span * geometric_to_seconds
window_end = -args.align_end_before * geometric_to_seconds
tau_max = 50.0 * geometric_to_seconds


def comparison_window(f_min):
    """
    (window, taper_length) for a given analysis f_min.

    The window has to start after the Stage-1 taper: that region is where the
    conditioning is *supposed* to change the waveform, so including it would be
    measuring the taper rather than the plugin. The taper length is the same
    quantity the source model hands to SimInspiralTDConditionStage1.
    """
    mass_1, mass_2, spin_1z, spin_2z = component_masses_and_spins(
        gra.metadata, args.total_mass
    )
    _, _, _, chirp_time, extra_time, fraction = compute_conditioning_parameters(
        dict(mass1=mass_1, mass2=mass_2, spin1z=spin_1z, spin2z=spin_2z, f_lower=f_min)
    )
    taper_length = fraction * chirp_time + extra_time
    start = record_start + taper_length + args.settle_cycles / f_min
    if start >= window_end:
        return None, taper_length
    return numpy.arange(start, window_end, 1.0 / args.srate), taper_length


window, taper_length = comparison_window(args.f_min)
if window is None:
    raise SystemExit(
        f"  the Stage-1 taper ({taper_length:.3f} s) leaves nothing of the "
        f"{-record_start:.3f} s record -- raise --total_mass or --f_min."
    )
print(
    f"  record starts {record_start:.3f} s before merger; Stage-1 taper is "
    f"{taper_length:.3f} s at f_min = {args.f_min:.0f} Hz\n"
    f"  comparison window: [{window[0]:+.3f}, {window[-1]:+.3f}] s "
    f"({window[-1] - window[0]:.3f} s usable)"
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: before vs after the conditioning
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 78)
print("Test 1: the same waveform, before and after the LAL conditioning")
print("=" * 78)


def roundtrip(f_min, window):
    """Same waveform in and out; the only thing acting is the conditioning."""
    kwargs = gra_arguments(cut_U=gra_kwargs["cut_U"], minimum_frequency=f_min)
    time, hplus, hcross = nr_time_domain_polarisations(
        args.total_mass, args.distance, args.iota, args.phase, **kwargs
    )
    conditioned = nr_frequency_domain_source_model(
        frequency_array,
        args.total_mass,
        args.distance,
        args.iota,
        args.phase,
        **kwargs,
    )
    result = difference_of_polarisations(
        window,
        segment_time,
        to_time_domain(conditioned["plus"]),
        to_time_domain(conditioned["cross"]),
        time,
        hplus,
        hcross,
        tau_max,
    )
    return time, hplus, hcross, conditioned, result


time_before, hplus_before, hcross_before, conditioned, roundtrip_result = roundtrip(
    args.f_min, window
)
hplus_after = to_time_domain(conditioned["plus"])
hcross_after = to_time_domain(conditioned["cross"])

# Nothing is band-limited by hand here. `after` carries whatever band the
# conditioning left it with, `before` is the untouched NR waveform, and the
# window was chosen so the instantaneous frequency is inside [f_min, f_max]
# throughout -- which is the only region where the two can be expected to agree.
frequency_in_window = numpy.interp(
    window,
    time_before,
    instantaneous_frequency(time_before, hplus_before, hcross_before),
)
print(
    f"\n  GW frequency over the window: "
    f"{frequency_in_window.min():.1f} - {frequency_in_window.max():.1f} Hz "
    f"(band is {args.f_min:.0f} - {args.f_max:.0f} Hz)"
)

print("\n  after vs before, over the comparison window")
print("  (tau should come out ~0 -- that is the epoch check. dphi is not")
print("   meaningful here: the two phases are unwrapped from different origins.)")
report("conditioned vs raw", roundtrip_result, unit=" s")

# The residual is set by how far the analysis f_min sits below the frequency at
# which the NR record starts. Above it, the high-pass acts on the whole waveform.
scan = [value for value in args.f_min_scan if value != args.f_min] + [args.f_min]
if len(scan) > 1:
    print(
        f"\n  the same test at other f_min "
        f"(the record starts at {gra_frequency / geometric_to_seconds:.1f} Hz):"
    )
    print(
        f"    {'f_min':>6s} {'taper':>8s} {'window':>16s} {'max|dA/A|':>11s} "
        f"{'max|dphi|':>11s}"
    )
    for f_min in sorted(scan):
        scan_window, scan_taper = comparison_window(f_min)
        if scan_window is None:
            print(f"    {f_min:6.0f} {scan_taper:8.3f} {'(nothing left)':>16s}")
            continue
        *_, scan_result = roundtrip(f_min, scan_window)
        print(
            f"    {f_min:6.0f} {scan_taper:8.3f} "
            f"{f'[{scan_window[0]:+.2f},{scan_window[-1]:+.2f}]':>16s} "
            f"{numpy.max(numpy.abs(scan_result['relative_amplitude'])):11.3e} "
            f"{numpy.max(numpy.abs(scan_result['phase_difference'])):11.3e}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: GRAthena++ vs SXS, raw and through the plugin
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 78)
print(f"Test 2: GRAthena:BHBH:{args.gra_id} vs SXS:BBH:{sxs_id}")
print("=" * 78)

# (a) raw: the (2,2) modes in geometric units, no conditioning anywhere. Same
# window as (b), expressed in M.
raw_window = numpy.arange(
    window[0] / geometric_to_seconds, window[-1] / geometric_to_seconds, 0.5
)
raw = difference(
    raw_window,
    gra.u - gra_peak,
    gra.hlm[(2, 2)]["A"],
    gra.hlm[(2, 2)]["p"],
    sxs.u - sxs_peak,
    sxs.hlm[(2, 2)]["A"],
    sxs.hlm[(2, 2)]["p"],
    50.0,
)
print("\n  (a) raw (2,2) modes, geometric units, no conditioning")
report("GRA vs SXS", raw, unit=" M")

# (b) through the source model.
gra_conditioned = nr_frequency_domain_source_model(
    frequency_array, args.total_mass, args.distance, args.iota, args.phase, **gra_kwargs
)
sxs_conditioned = nr_frequency_domain_source_model(
    frequency_array, args.total_mass, args.distance, args.iota, args.phase, **sxs_kwargs
)
plugin = difference_of_polarisations(
    window,
    segment_time,
    to_time_domain(gra_conditioned["plus"]),
    to_time_domain(gra_conditioned["cross"]),
    segment_time,
    to_time_domain(sxs_conditioned["plus"]),
    to_time_domain(sxs_conditioned["cross"]),
    tau_max,
)
print("\n  (b) through the source model, conditioned and band-limited")
report("GRA vs SXS", plugin, unit=" s")

# Put the raw difference on the plugin's grid so the two can be subtracted.
raw_seconds = raw_window * geometric_to_seconds
raw_phase = numpy.interp(
    window, raw_seconds, raw["phase_difference"], left=numpy.nan, right=numpy.nan
)
raw_amplitude = numpy.interp(
    window, raw_seconds, raw["relative_amplitude"], left=numpy.nan, right=numpy.nan
)
overlap = numpy.isfinite(raw_phase)
phase_discrepancy = numpy.max(
    numpy.abs(plugin["phase_difference"][overlap] - raw_phase[overlap])
)
amplitude_discrepancy = numpy.max(
    numpy.abs(plugin["relative_amplitude"][overlap] - raw_amplitude[overlap])
)
print("\n  (b) - (a): what the conditioning added on top of the NR difference")
print(
    f"      max |dphi_plugin - dphi_raw| = {phase_discrepancy:.3e} rad"
    f"   (max |dphi_raw| = {numpy.max(numpy.abs(raw['phase_difference'])):.3e})"
)
print(
    f"      max |dA/A_plugin - dA/A_raw| = {amplitude_discrepancy:.3e}"
    f"   (max |dA/A_raw| = {numpy.max(numpy.abs(raw['relative_amplitude'])):.3e})"
)

if args.no_plot:
    raise SystemExit(0)

# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

figure, axes = plt.subplots(4, 1, figsize=(9, 12))
axes[0].plot(time_before, hplus_before, label="before conditioning")
axes[0].plot(segment_time, hplus_after, "--", label="after conditioning")
axes[0].axvspan(
    time_before[0], window[0], color="0.85", zorder=0, label="taper / excluded"
)
axes[0].set_xlim(time_before[0], time_before[-1])
axes[0].set_ylabel(r"$h_+(t)$")
axes[0].set_xlabel("time to merger [s]")
axes[0].legend(fontsize=8)

axes[1].semilogy(
    window, numpy.abs(roundtrip_result["relative_amplitude"]), label=r"$|\Delta A / A|$"
)
axes[1].semilogy(
    window, roundtrip_result["relative_strain"], label=r"$|\Delta h| / |h|$"
)
axes[1].set_ylabel("relative difference")
axes[1].set_xlabel("time to merger [s]")
axes[1].legend()

axes[2].plot(window, roundtrip_result["phase_difference"])
axes[2].axhline(0.0, color="k", lw=0.5)
axes[2].set_ylabel(r"$\Delta \phi$ [rad]")
axes[2].set_xlabel("time to merger [s]")

axes[3].plot(window, frequency_in_window)
axes[3].axhline(args.f_min, color="r", lw=0.8, ls="--", label=r"$f_{\min}$")
axes[3].set_ylabel("GW frequency [Hz]")
axes[3].set_xlabel("time to merger [s]")
axes[3].legend()

figure.suptitle(
    f"Conditioning round trip, GRAthena:BHBH:{args.gra_id}, "
    f"M = {args.total_mass:.0f} $M_\\odot$"
)
figure.tight_layout()
roundtrip_file = os.path.join(out_path, f"conditioning_roundtrip_{args.gra_id}.png")
figure.savefig(roundtrip_file, dpi=120)

figure, axes = plt.subplots(3, 1, figsize=(9, 10))
axes[0].semilogy(gra.u - gra_peak, gra.hlm[(2, 2)]["A"], label=f"GRA:{args.gra_id}")
axes[0].semilogy(
    sxs.u - sxs_peak, sxs.hlm[(2, 2)]["A"], "--", label=f"SXS:BBH:{sxs_id}"
)
axes[0].axvline(
    args.t_after_peak,
    color="r",
    ls="--",
    lw=0.8,
    label=f"trim at +{args.t_after_peak:.0f} M",
)
axes[0].set_ylabel(r"$A_{22}$")
axes[0].set_xlabel(r"$u - u_{\rm peak}$ [M]")
axes[0].legend(fontsize=8)

axes[1].plot(raw_seconds, raw["relative_amplitude"], label="raw modes")
axes[1].plot(window, plugin["relative_amplitude"], "--", label="through the plugin")
axes[1].axhline(0.0, color="k", lw=0.5)
axes[1].set_ylabel(r"$\Delta A / A$")
axes[1].set_xlabel("time to merger [s]")
axes[1].legend()

axes[2].plot(raw_seconds, raw["phase_difference"], label="raw modes")
axes[2].plot(window, plugin["phase_difference"], "--", label="through the plugin")
axes[2].axhline(0.0, color="k", lw=0.5)
axes[2].set_ylabel(r"$\Delta \phi$ [rad]")
axes[2].set_xlabel("time to merger [s]")
axes[2].legend()

figure.suptitle(
    f"GRAthena:BHBH:{args.gra_id} vs SXS:BBH:{sxs_id}, "
    f"q = {gra.metadata['q']:.2f}, M = {args.total_mass:.0f} $M_\\odot$"
)
figure.tight_layout()
cross_code_file = os.path.join(out_path, f"nr_cross_code_{args.gra_id}.png")
figure.savefig(cross_code_file, dpi=120)

print(f"\nsaved {roundtrip_file}")
print(f"saved {cross_code_file}")
