"""
Bilby frequency-domain source models backed by PyART's NR catalogues.
Usage:
    import bilby
    from PyART.plugin.bilby_plugin import nr_frequency_domain_source_model

    waveform_arguments = dict(
        catalog="sxs", ID="0305", path="./local_data/sxs/", download=True,
        ellmax=4, cut_U=200,
        minimum_frequency=20.0, maximum_frequency=1024.0,
        sampling_frequency=4096.0,
    )
    generator = bilby.gw.WaveformGenerator(
        duration=4.0, sampling_frequency=4096.0,
        frequency_domain_source_model=nr_frequency_domain_source_model,
        waveform_arguments=waveform_arguments,
    )
"""

import logging
import numpy
from itertools import product
from scipy.interpolate import CubicSpline

try:
    import lal
    import lalsimulation
except ImportError:
    raise ImportError("WARNING: LALSimulation and/or LAL not installed.")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared conditioning-parameter computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_conditioning_parameters(binary_parameters):
    """
    This conditoning scheme is to closely approximate SimInspiralFD-style. Much of it is ~~copied~~ inspired by
    https://github.com/AEI-ACR/pyseobnr/blob/70e730002061210264dcc297d01e6b4c8c5a0bc1/pyseobnr/generate_waveform.py#L1258

    Returns
    -------
    f_start : float
        Lower starting frequency for the TD waveform call (includes extra time).
    f_lower : float
        Effective minimum frequency used for the chirp-time bound; equal to
        ``min(minimum_frequency, ISCO_9M)`` following the SimInspiralFD convention.
    fico : float
        Schwarzschild ISCO frequency used for Stage 2.
    chirp_time : float
        Chirp time at ``f_lower``.
    extra_time : float
        Extra time (extra_cycles / f_lower) prepended before the waveform.
    extra_time_fraction : float
        Fraction of ``chirp_time`` used for the Stage-1 taper (0.1).
    """
    extra_time_fraction = 0.1
    extra_cycles = 3.0
    mass1 = binary_parameters["mass1"]
    mass2 = binary_parameters["mass2"]
    spin1z = binary_parameters["spin1z"]
    spin2z = binary_parameters["spin2z"]
    minimum_frequency = binary_parameters["f_lower"]

    # SimInspiralFD uses r = 9M ISCO to decide whether to lower f_min.
    fico_check = 1.0 / (pow(9.0, 1.5) * numpy.pi * (mass1 + mass2) * lal.MTSUN_SI)
    f_lower = min(minimum_frequency, fico_check)

    # Schwarzschild (r = 6M) ISCO for Stage 2.
    fico = 1.0 / (pow(6.0, 1.5) * numpy.pi * (mass1 + mass2) * lal.MTSUN_SI)

    # chirp time
    chirp_time = lalsimulation.SimInspiralChirpTimeBound(
        f_lower, mass1 * lal.MSUN_SI, mass2 * lal.MSUN_SI, spin1z, spin2z
    )

    spinkerr = lalsimulation.SimInspiralFinalBlackHoleSpinBound(spin1z, spin2z)
    tmerge = lalsimulation.SimInspiralMergeTimeBound(
        mass1 * lal.MSUN_SI, mass2 * lal.MSUN_SI
    ) + lalsimulation.SimInspiralRingdownTimeBound(
        (mass1 + mass2) * lal.MSUN_SI, spinkerr
    )

    extra_time = extra_cycles / f_lower
    f_start = lalsimulation.SimInspiralChirpStartFrequencyBound(
        (1.0 + extra_time_fraction) * chirp_time + tmerge + extra_time,
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
    )

    return f_start, f_lower, fico, chirp_time, extra_time, extra_time_fraction


def get_frequency_domain_polarisations(
    hp_ts, hc_ts, sampling_frequency, delta_f, maximum_frequency=None
):
    """
    Resize conditioned TD polarisations to a power-of-2 chirp length and
    FFT with LAL's REAL8TimeFreqFFT.

    Again inspired by this: https://github.com/AEI-ACR/pyseobnr/blob/70e730002061210264dcc297d01e6b4c8c5a0bc1/pyseobnr/generate_waveform.py#L1333

    Returns
    -------
    frequency_array : ndarray  -- one-sided frequencies [0 .. Nyquist].
    hptilde : frequency domain h+(f) as LAL COMPLEX16FrequencySeries (one-sided, up to Nyquist).
    hctilde : frequency domain hx(f) as LAL COMPLEX16FrequencySeries (one-sided, up to Nyquist).
    """
    nyquist = sampling_frequency // 2
    if maximum_frequency is None:
        maximum_frequency = nyquist

    if delta_f != 0:
        n = int(numpy.round(maximum_frequency / delta_f))
        if n & (n - 1):
            exp = numpy.frexp(n)
            nyquist = numpy.ldexp(1, int(exp[1])) * delta_f

    delta_t = 0.5 / nyquist

    if delta_f == 0:
        chirp_length = hp_ts.data.length
        exp = numpy.frexp(chirp_length)
        chirp_length = int(numpy.ldexp(1, int(exp[1])))
        delta_f = 1.0 / (chirp_length * delta_t)
    else:
        chirp_length = int(1.0 / (delta_f * delta_t))

    # Keep the last chirp_length samples (i.e. keep the merger / ringdown).
    # When the NR waveform is shorter than the segment the first argument is
    # negative and LAL prepends zeros, which is what we want.
    lal.ResizeREAL8TimeSeries(hp_ts, hp_ts.data.length - chirp_length, chirp_length)
    lal.ResizeREAL8TimeSeries(hc_ts, hc_ts.data.length - chirp_length, chirp_length)

    hptilde = lal.CreateCOMPLEX16FrequencySeries(
        "FD H_PLUS",
        hp_ts.epoch,
        0.0,
        delta_f,
        lal.DimensionlessUnit,
        int(chirp_length / 2 + 1),
    )

    hctilde = lal.CreateCOMPLEX16FrequencySeries(
        "FD H_CROSS",
        hc_ts.epoch,
        0.0,
        delta_f,
        lal.DimensionlessUnit,
        int(chirp_length / 2 + 1),
    )

    plan = lal.CreateForwardREAL8FFTPlan(chirp_length, 0)
    lal.REAL8TimeFreqFFT(hptilde, hp_ts, plan)
    lal.REAL8TimeFreqFFT(hctilde, hc_ts, plan)

    frequency_array = numpy.arange(len(hptilde.data.data)) * hptilde.deltaF
    return frequency_array, hptilde, hctilde


# ─────────────────────────────────────────────────────────────────────────────
# NR-specific helpers
# ─────────────────────────────────────────────────────────────────────────────


def default_mode_array(ellmax, load_m0=False):
    """
    The (l, m) list that PyART's catalog loaders build, repeated here so the
    caller can ask for a subset without reaching into them.
    """
    return [
        (l, m)
        for l, m in product(range(2, ellmax + 1), range(-ellmax, ellmax + 1))
        if (m != 0 or load_m0) and l >= numpy.abs(m)
    ]


def invariant_amplitude(hlm, mode_array):
    """
    sqrt(sum_lm |h_lm|^2): the frame-invariant amplitude, whose peak is what LAL
    call t = 0.
    """
    amplitude = 0.0
    for mode in mode_array:
        amplitude = amplitude + numpy.abs(hlm[tuple(mode)]["A"]) ** 2
    return numpy.sqrt(amplitude)


def estimate_time_of_maximum_amplitude(time_array, hlm, mode_array, precision=1e-3):
    """
    identical to what is done in pyseobnr
    """
    invariant_amp = invariant_amplitude(hlm, mode_array)
    assert len(invariant_amp) == len(time_array)
    amplitude_interpolator = CubicSpline(time_array, invariant_amp)

    index_of_peak_amplitude = int(numpy.argmax(invariant_amp))
    coarse_peak_time = time_array[index_of_peak_amplitude]

    # local spacing: the grid need not be uniform (SXS output is adaptive)
    i = min(index_of_peak_amplitude, len(time_array) - 2)
    delta_t = time_array[i + 1] - time_array[i]

    step = min(precision, delta_t / 10.0)
    fine_peak_time = numpy.arange(
        coarse_peak_time - delta_t, coarse_peak_time + delta_t, step
    )
    amplitude_fine = amplitude_interpolator(fine_peak_time)
    return float(fine_peak_time[numpy.argmax(amplitude_fine)])


def trim_nr_junk_and_drift(u, hlm, mode_array, peak_u, t_final=None, t_after_peak=None):
    """
    Drop the tail of the NR record.

    GRAthena++ ringdowns drift at late times.
    Two ways to cut it, both in geometric units (M):

    t_final      : absolute cut in the catalogue's own retarded time u.
    t_after_peak : cut this many M after the invariant-amplitude peak. Anchored
                   to the merger rather than to the end of the file, so it
                   survives a change of cut_U or resolution.

    If both are given, the earlier of the two wins!

    Returns
    -------
    u, hlm : the trimmed time array and mode dict (a new dict; the input, which
             is owned by the cached Waveform object, is not modified).
    """
    keep = numpy.ones(len(u), dtype=bool)
    if t_final is not None:
        keep &= u <= t_final
    if t_after_peak is not None:
        keep &= u <= peak_u + t_after_peak

    if keep.all():
        return u, hlm

    if not keep.any():
        raise RuntimeError(
            f"t_final={t_final} / t_after_peak={t_after_peak} removed the whole "
            f"waveform (u spans [{u[0]:.1f}, {u[-1]:.1f}], peak at {peak_u:.1f})"
        )

    logger.debug(
        "trimming NR data at u <= %.1f M (was %.1f M): %d of %d samples kept",
        u[keep][-1],
        u[-1],
        keep.sum(),
        len(u),
    )
    trimmed = {
        tuple(mode): {k: v[keep] for k, v in hlm[tuple(mode)].items()}
        for mode in mode_array
    }
    return u[keep], trimmed


def compute_polarisations_from_modes(hlm, mode_array, iota, phase):
    signal = 0.0 + 0.0j
    for mode in mode_array:
        ell, emm = int(mode[0]), int(mode[1])
        ylm = lal.SpinWeightedSphericalHarmonic(
            iota, numpy.pi / 2 - phase, -2, ell, emm
        )
        signal = signal + ylm * hlm[(ell, emm)]
    return numpy.real(signal), -numpy.imag(signal)


def interpolate_modes(u, hlm, mode_array, new_u):
    """
    Returns a plain {(l, m): complex ndarray} dict.
    """
    from scipy.interpolate import CubicSpline

    out = {}
    for mode in mode_array:
        key = tuple(mode)
        # Adopted Cubic spline rather than what PyART does here:
        # https://github.com/RoxGamba/PyART/blob/05bb78ff58b73f0a2480fdbcbc010d0cf71d950f/PyART/waveform.py#L502
        # amplitude = numpy.interp(new_u, u, hlm[key]["A"])
        # phase = numpy.interp(new_u, u, hlm[key]["p"])
        amplitude = CubicSpline(u, hlm[key]["A"])(new_u)
        phase = CubicSpline(u, hlm[key]["p"])(new_u)
        out[key] = amplitude * numpy.exp(-1j * phase)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Catalogue loading (memoised: the likelihood must not touch the disk)
# ─────────────────────────────────────────────────────────────────────────────

_WAVEFORM_CACHE = {}

# Per-catalogue defaults for the late-time trim. GRAthena++ ringdowns drift, so
# that catalogue gets a merger-anchored cut by default; SXS does not need one.
_DEFAULT_T_AFTER_PEAK = {"sxs": None, "gra": 100.0}


def load_nr_waveform(**waveform_arguments):
    """
    Return a PyART Waveform for the requested simulation, loading it at most once.
    """
    catalog = str(waveform_arguments["catalog"]).lower()
    ID = waveform_arguments["ID"]
    if isinstance(ID, int):
        ID = f"{ID:04}"
    path = waveform_arguments["path"]
    ellmax = waveform_arguments.get("ellmax", 4)
    cut_U = waveform_arguments.get("cut_U", None)
    load_m0 = waveform_arguments.get("load_m0", True)
    download = waveform_arguments.get("download", False)

    if catalog == "sxs":
        order = waveform_arguments.get("order", 2)
        level = waveform_arguments.get("level", None)
        key = ("sxs", ID, path, ellmax, cut_U, load_m0, order, level)
    elif catalog == "gra":
        ext = waveform_arguments.get("ext", "ext")
        res = waveform_arguments.get("res", "128")
        r_ext = waveform_arguments.get("r_ext", None)
        key = ("gra", ID, path, ellmax, cut_U, load_m0, ext, res, r_ext)
    else:
        raise NotImplementedError(
            f"Not implemented '{catalog}'. "
            f"Ask someone to add support for catalog '{catalog}' or force Koustav to do it"
        )

    if key in _WAVEFORM_CACHE:
        return _WAVEFORM_CACHE[key]

    logger.info("loading %s simulation %s from %s", catalog.upper(), ID, path)
    if catalog.lower() == "sxs":
        from ..catalogs import sxs as sxs_catalog

        waveform = sxs_catalog.Waveform_SXS(
            path=path,
            ID=ID,
            order=order,
            level=level,
            ellmax=ellmax,
            cut_U=cut_U,
            load_m0=load_m0,
            download=download,
            nu_rescale=False,
            load=["hlm", "metadata"],
            ignore_deprecation=waveform_arguments.get("ignore_deprecation", True),
        )
    elif catalog.lower() == "gra":
        from ..catalogs import gra as gra_catalog

        waveform = gra_catalog.Waveform_GRA(
            ID=ID,
            path=path,
            ellmax=ellmax,
            ext=ext,
            res=res,
            r_ext=r_ext,
            cut_U=cut_U,
            download=download,
            nu_rescale=False,
        )
    else:
        raise ValueError(
            f'Ask someone to add support for catalog "{catalog}" '
            f"or force Koustav to do it"
        )

    _WAVEFORM_CACHE[key] = waveform
    return waveform


def clear_waveform_cache():
    """Drop every memoised simulation. Mostly useful in tests."""
    _WAVEFORM_CACHE.clear()


def component_masses_and_spins(metadata, total_mass):
    """
    Who is fatter?
    """
    m1, m2 = float(metadata["m1"]), float(metadata["m2"])
    mass_1 = total_mass * m1 / (m1 + m2)
    mass_2 = total_mass * m2 / (m1 + m2)
    spin_1x, spin_1y, spin_1z = (
        float(metadata["chi1x"]),
        float(metadata["chi1y"]),
        float(metadata["chi1z"]),
    )
    spin_2x, spin_2y, spin_2z = (
        float(metadata["chi2x"]),
        float(metadata["chi2y"]),
        float(metadata["chi2z"]),
    )
    if mass_1 < mass_2:
        mass_1, mass_2 = mass_2, mass_1
        spin_1x, spin_2x = spin_2x, spin_1x
        spin_1y, spin_2y = spin_2y, spin_1y
        spin_1z, spin_2z = spin_2z, spin_1z
    return mass_1, mass_2, spin_1z, spin_2z


def resolve_mode_array(waveform, waveform_arguments):
    """The modes to sum: everything that was loaded, or the requested subset."""
    mode_array = waveform_arguments.get("mode_array", None)
    if mode_array is None:
        return sorted(waveform.hlm.keys())
    mode_array = [tuple(mode) for mode in mode_array]
    missing = [mode for mode in mode_array if mode not in waveform.hlm]
    if missing:
        raise KeyError(f"modes {missing} were not loaded from the simulation")
    return mode_array


# ─────────────────────────────────────────────────────────────────────────────
# Bilby source models
# ─────────────────────────────────────────────────────────────────────────────


def nr_time_domain_polarisations(
    total_mass,
    luminosity_distance,
    iota,
    phase,
    **waveform_arguments,
):
    """
    The NR polarisations in SI units, on a uniform grid, with t = 0 at the peak
    of the invariant amplitude -- and *before* any of the LAL conditioning.

    Returns
    -------
    time : ndarray
        Seconds, uniformly spaced, negative before the merger.
    hplus, hcross : ndarray
        Dimensionless strain at ``luminosity_distance``.
    """
    sampling_frequency = waveform_arguments["sampling_frequency"]

    waveform = load_nr_waveform(**waveform_arguments)
    catalog = str(waveform_arguments["catalog"]).lower()
    mode_array = resolve_mode_array(waveform, waveform_arguments)

    geometric_to_seconds = total_mass * lal.MTSUN_SI
    distance_in_metres = luminosity_distance * 1e6 * lal.PC_SI

    u = numpy.asarray(waveform.u)
    # Centre t = 0 on the peak of the invariant amplitude, which is where LAL
    # put it, then drop the late-time tail if requested.
    peak_u = estimate_time_of_maximum_amplitude(u, waveform.hlm, mode_array)
    t_after_peak = waveform_arguments.get(
        "t_after_peak", _DEFAULT_T_AFTER_PEAK[catalog]
    )
    u, hlm = trim_nr_junk_and_drift(
        u,
        waveform.hlm,
        mode_array,
        peak_u,
        t_final=waveform_arguments.get("t_final", None),
        t_after_peak=t_after_peak,
    )

    times = (u - peak_u) * geometric_to_seconds

    delta_t = 1.0 / sampling_frequency
    time = numpy.arange(times[0], times[-1], delta_t)
    hlm_interpolated = interpolate_modes(times, hlm, mode_array, time)

    hplus, hcross = compute_polarisations_from_modes(
        hlm_interpolated, mode_array, iota, phase
    )
    amplitude_scale = geometric_to_seconds * lal.C_SI / distance_in_metres
    return time, hplus * amplitude_scale, hcross * amplitude_scale


def nr_frequency_domain_source_model(
    frequency_array,
    total_mass,
    luminosity_distance,
    iota,
    phase,
    **waveform_arguments,
):
    """
    A bilby frequency_domain_source_model backed by an NR simulation.

    Parameters
    ----------
    frequency_array : ndarray
        Bilby's frequency grid.
    total_mass : float
        Total mass of the binary in solar masses.
    luminosity_distance : float
        Luminosity distance in Mpc.
    iota : float
        Inclination angle between the orbital angular momentum and the line of sight.
    phase : float
        Orbital phase. Note that phase = 0 does not single out any particular
        orientation of the simulation: an NR run carries its own arbitrary
        orbital phase at the reference time, so there is a constant offset
        between this parameter and LAL's phiRef. That is harmless when phase is
        sampled, but it means a fixed-phase comparison against an approximant
        has to search over it -- with only the (2, +-2) modes the offset is
        degenerate with the overall phase, but with higher modes it is not.
    waveform_arguments : dict
        Required: ``catalog``, ``ID``, ``path``, ``minimum_frequency``,
        ``maximum_frequency``, ``sampling_frequency``.
        Optional: ``ellmax`` (4), ``load_m0`` (True), ``cut_U``, ``t_final``,
        ``t_after_peak``, ``mode_array``, ``download``, plus the
        catalogue-specific keys accepted by :func:`load_nr_waveform`.

    Returns
    -------
    dict with keys 'plus' and 'cross'.
    """
    minimum_frequency = waveform_arguments["minimum_frequency"]
    maximum_frequency = waveform_arguments["maximum_frequency"]
    sampling_frequency = waveform_arguments["sampling_frequency"]
    delta_f = frequency_array[1] - frequency_array[0]

    waveform = load_nr_waveform(**waveform_arguments)
    mass_1, mass_2, spin_1z, spin_2z = component_masses_and_spins(
        waveform.metadata, total_mass
    )

    binary_parameters = {
        "mass1": mass_1,
        "mass2": mass_2,
        "spin1z": spin_1z,
        "spin2z": spin_2z,
        "f_lower": minimum_frequency,
    }
    _, _, fico, chirp_time, extra_time, extra_time_fraction = (
        compute_conditioning_parameters(binary_parameters)
    )

    time, hplus, hcross = nr_time_domain_polarisations(
        total_mass, luminosity_distance, iota, phase, **waveform_arguments
    )
    delta_t = 1.0 / sampling_frequency
    lal_epoch = lal.LIGOTimeGPS(float(time[0]))

    # SimInspiralFD generates from a frequency low enough that the taper below
    # has somewhere to act. An NR record is however long it is, and if it is
    # shorter than the taper, SimInspiralTDConditionStage1 aborts the process
    # rather than returning an error -- so check here, where we can still say
    # something useful.
    taper_length = extra_time_fraction * chirp_time + extra_time
    record_length = float(time[-1] - time[0])
    if record_length <= taper_length:
        raise ValueError(
            f"the NR record is {record_length:.3f} s long at M = {total_mass:.1f} "
            f"Msun, but the SimInspiralFD conditioning wants to taper "
            f"{taper_length:.3f} s of it at minimum_frequency = "
            f"{minimum_frequency:.1f} Hz. Raise the total mass, raise "
            f"minimum_frequency, or use a longer simulation."
        )

    # Same treatment before Fourier transforming as SimInspiralFD applies.
    # Stage-1: taper the initial extra segment before f_min, high-pass at f_min
    # and remove zero-padding. Stage-2: taper one cycle at f_min from the start
    # and one at f_ISCO from the end, to kill the filter transients.
    # https://github.com/lscsoft/lalsuite/blob/06047f8491f5cf480c720b47e2ad7bf85b26adc4/lalsimulation/lib/LALSimInspiral.c#L5504
    time_length = len(hplus)
    hp_ts = lal.CreateREAL8TimeSeries(
        "hplus", lal_epoch, 0, delta_t, lal.DimensionlessUnit, time_length
    )
    hc_ts = lal.CreateREAL8TimeSeries(
        "hcross", lal_epoch, 0, delta_t, lal.DimensionlessUnit, time_length
    )
    hp_ts.data.data[:] = hplus
    hc_ts.data.data[:] = hcross

    lalsimulation.SimInspiralTDConditionStage1(
        hp_ts, hc_ts, extra_time_fraction * chirp_time + extra_time, minimum_frequency
    )
    lalsimulation.SimInspiralTDConditionStage2(hp_ts, hc_ts, minimum_frequency, fico)

    _, hptilde, hctilde = get_frequency_domain_polarisations(
        hp_ts, hc_ts, sampling_frequency, delta_f, maximum_frequency=maximum_frequency
    )

    hplus_output = numpy.zeros_like(frequency_array, dtype=complex)
    hcross_output = numpy.zeros_like(frequency_array, dtype=complex)
    n = len(frequency_array)
    if len(hptilde.data.data) > n:
        hplus_output = hptilde.data.data[:n].copy()
        hcross_output = hctilde.data.data[:n].copy()
    else:
        hplus_output[: len(hptilde.data.data)] = hptilde.data.data
        hcross_output[: len(hctilde.data.data)] = hctilde.data.data

    frequency_bounds = (frequency_array >= minimum_frequency) & (
        frequency_array <= maximum_frequency
    )
    hplus_output *= frequency_bounds
    hcross_output *= frequency_bounds

    # Time shift: move the merger to t = 0, exactly as bilby does on the
    # SimInspiralFD branch of _base_lal_cbc_fd_waveform.
    dt = (
        1.0 / hptilde.deltaF
        + hptilde.epoch.gpsSeconds
        + hptilde.epoch.gpsNanoSeconds * 1e-9
    )
    time_shift = numpy.exp(-2j * numpy.pi * frequency_array[frequency_bounds] * dt)
    hplus_output[frequency_bounds] *= time_shift
    hcross_output[frequency_bounds] *= time_shift

    return dict(plus=hplus_output, cross=hcross_output)


def sxs_frequency_domain_source_model(
    frequency_array,
    total_mass,
    luminosity_distance,
    iota,
    phase,
    **waveform_arguments,
):
    """:func:`nr_frequency_domain_source_model` pinned to the SXS catalogue."""
    waveform_arguments["catalog"] = "sxs"
    return nr_frequency_domain_source_model(
        frequency_array,
        total_mass,
        luminosity_distance,
        iota,
        phase,
        **waveform_arguments,
    )


def gra_frequency_domain_source_model(
    frequency_array,
    total_mass,
    luminosity_distance,
    iota,
    phase,
    **waveform_arguments,
):
    """:func:`nr_frequency_domain_source_model` pinned to the GRAthena++ catalogue."""
    waveform_arguments["catalog"] = "gra"
    return nr_frequency_domain_source_model(
        frequency_array,
        total_mass,
        luminosity_distance,
        iota,
        phase,
        **waveform_arguments,
    )
