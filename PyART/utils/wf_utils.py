import numpy as np
from scipy.optimize import minimize_scalar

from . import utils as ut


def mnfactor(m):
    """
    Factor to account for negative m modes
    """
    return 1 if m == 0 else 2


# Various multipolar coefficients (mc) needed below
def mc_f(l, m):
    return np.sqrt(l * (l + 1) - m * (m + 1))


def mc_a(l, m):
    return np.sqrt((l - m) * (l + m + 1)) / (l * (l + 1))


def mc_b(l, m):
    return np.sqrt(
        ((l - 2) * (l + 2) * (l + m) * (l + m - 1)) / ((2 * l - 1) * (2 * l + 1))
    ) / (2 * l)


def mc_c(l, m):
    return 2 * m / (l * (l + 1))


def mc_d(l, m):
    return (
        np.sqrt(((l - 2) * (l + 2) * (l - m) * (l + m)) / ((2 * l - 1) * (2 * l + 1)))
        / l
    )


def mode_to_k(ell, emm):
    return int(ell * (ell - 1) / 2 + emm - 2)


def modes_to_k(modes):
    return [mode_to_k(x[0], x[1]) for x in modes]


def k_to_ell(k):
    LINDEX = [
        2,
        2,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
    ]
    return LINDEX[k]


def k_to_emm(k):
    MINDEX = [
        1,
        2,
        1,
        2,
        3,
        1,
        2,
        3,
        4,
        1,
        2,
        3,
        4,
        5,
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ]
    return MINDEX[k]


def invariant_amplitude(hlm, modes=None):
    """
    Frame-invariant amplitude sqrt(sum_lm |h_lm|^2).

    Unlike the amplitude of a single multipole this does not depend on the
    orientation of the frame, which is why its peak is the time origin used by
    LAL (and hence by the NR source models).

    Parameters
    ----------
    hlm: dict
        dictionary with multipoles
    modes: list or None
        list of (ell, emm) tuples to include. If None, use every mode in hlm.
    Returns
    -------
    out: ndarray
        frame-invariant amplitude
    """
    if modes is None:
        modes = list(hlm.keys())
    amplitude = 0.0
    for k in modes:
        amplitude = amplitude + hlm[tuple(k)]["A"] ** 2
    return np.sqrt(amplitude)


def compute_hphc(hlm, phi=0, i=0, modes=[(2, 2)], assume_symmetry=True):
    """
    Compute hp and hc from hlm

    Parameters
    ----------
    hlm: dict
        dictionary with multipoles
    phi: float
        azimuthal angle
    i: float
        inclination angle
    modes: list
        list of (ell, emm) tuples
    assume_symmetry: bool
        if True (default), assume the usual symmetry between hlm and hl-m, valid
        for aligned spins: only the m>0 modes need to be passed, and the m<0
        ones are reconstructed from them.
        if False, sum the requested modes as they are, with no reconstruction.
        This is the general case (e.g. precessing systems, or NR data where the
        m<0 modes are loaded independently), but the caller is then responsible
        for passing BOTH the m>0 and the m<0 modes -- passing only (2,2) yields
        a circularly polarized waveform, not the physical pair.
    Returns
    -------
    out: (hp, hc)
        plus and cross polarizations
    """
    h = 0 + 1j * 0
    if assume_symmetry:
        for k in modes:
            ell = k[0]
            emm = k[1]
            if emm != 0:
                Alm = hlm[k]["A"]
            else:
                Alm = hlm[k]["z"] / 2
            plm = hlm[k]["p"]
            Hp = Alm * np.exp(-1j * plm)
            Hn = (-1) ** ell * Alm * np.exp(1j * plm)
            Ylmp = ut.spinsphericalharm(-2, ell, emm, np.pi / 2 - phi, i)
            Ylmn = ut.spinsphericalharm(-2, ell, -emm, np.pi / 2 - phi, i)
            h += Ylmp * Hp + Ylmn * Hn
    else:
        for k in modes:
            ell = k[0]
            emm = k[1]
            Ylm = ut.spinsphericalharm(-2, ell, emm, np.pi / 2 - phi, i)
            h += Ylm * hlm[tuple(k)]["z"]

    hp = np.real(h)
    hc = -np.imag(h)
    return hp, hc


def get_multipole_dict(wave):
    """
    Given a complex waveform, return a dictionary with
    real, imag, z, A, p
    Parameters
    ----------
    wave: ndarray
        complex waveform
    Returns
    -------
    out: dict
        dictionary with real, imag, z, A, p
    """
    return {
        "real": wave.real,
        "imag": wave.imag,
        "z": wave,
        "A": np.abs(wave),
        "p": -np.unwrap(np.angle(wave)),
    }


##########################
#   Phasing, from watpy  #
##########################


def align_phase(t, Tf, phi_a_tau, phi_b):
    r"""
    Align two waveforms in phase by minimizing the chi^2

    \chi^2 = \int_0^Tf [\phi_a(t + \tau) - phi_b(t) - \Delta\phi]^2 dt

    as a function of \Delta\phi.

    * t         : time, must be equally spaced
    * Tf        : final time
    * phi_a_tau : time-shifted first phase evolution
    * phi_b     : second phase evolution

    This function returns \Delta\phi.

    Parameters
    ----------
    t: ndarray
        time array
    Tf: float
        final time
    phi_a_tau: ndarray
        first phase evolution, time-shifted
    phi_b: ndarray
        second phase evolution
    Returns
    -------
    out: float
        optimal phase shift
    """
    dt = t[1] - t[0]
    weight = np.double((t >= 0) & (t < Tf))
    return np.sum(weight * (phi_a_tau - phi_b) * dt) / np.sum(weight * dt)


def Align(t, Tf, tau_max, t_a, phi_a, t_b, phi_b, refine=False, tau_tol=1e-12):
    r"""
    Align two waveforms in phase by minimizing the chi^2

    chi^2 = \sum_{t_i=0}^{t_i < Tf} [phi_a(t_i + tau) - phi_b(t_i) - dphi]^2 dt

    as a function of dphi and tau.

    * t          : time
    * Tf         : final time
    * tau_max    : maximum time shift
    * t_a, phi_a : first phase evolution
    * t_b, phi_b : second phase evolution

    The two waveforms are re-sampled using the given time t

    This function returns a tuple (tau_opt, dphi_opt, chi2_opt)

    Parameters
    ----------
    t: ndarray
        time array
    Tf: float
        final time
    tau_max: float
        maximum time shift
    t_a: ndarray
        time array for first phase evolution
    phi_a: ndarray
        first phase evolution
    t_b: ndarray
        time array for second phase evolution
    phi_b: ndarray
        second phase evolution
    refine: bool
        if True, polish the best grid tau with a bounded scalar minimization
        over one grid spacing on either side, so the result is not limited to
        the sampling of t. The grid scan still runs first, which keeps the
        search global: minimizing directly over the full (-tau_max, tau_max)
        can settle in a 2*pi-aliased local minimum.
    tau_tol: float
        absolute tolerance on tau for the refinement
    Returns
    -------
    out: (tau_opt, dphi_opt, chi2_opt)
        optimal time shift, optimal phase shift, minimum chi^2
    """
    dt = t[1] - t[0]
    N = int(tau_max / dt)
    weight = np.double((t >= 0) & (t < Tf))

    res_phi_b = np.interp(t, t_b, phi_b)

    def chi2_of_tau(tau_value):
        res_phi_a_tau = np.interp(t, t_a + tau_value, phi_a)
        dphi_value = align_phase(t, Tf, res_phi_a_tau, res_phi_b)
        chi2_value = np.sum(weight * (res_phi_a_tau - res_phi_b - dphi_value) ** 2) * dt
        return chi2_value, dphi_value

    tau = []
    dphi = []
    chi2 = []
    for i in range(-N, N):
        tau.append(i * dt)
        chi2_value, dphi_value = chi2_of_tau(tau[-1])
        dphi.append(dphi_value)
        chi2.append(chi2_value)

    chi2 = np.array(chi2)
    imin = np.argmin(chi2)

    if not refine:
        return tau[imin], dphi[imin], chi2[imin]

    solution = minimize_scalar(
        lambda x: chi2_of_tau(x)[0],
        bounds=(tau[imin] - dt, tau[imin] + dt),
        method="bounded",
        options={"xatol": tau_tol},
    )
    tau_opt = float(solution.x)
    chi2_opt, dphi_opt = chi2_of_tau(tau_opt)
    # the scan already gives the global basin; keep it if the polish did worse
    if chi2_opt > chi2[imin]:
        return tau[imin], dphi[imin], chi2[imin]
    return tau_opt, dphi_opt, chi2_opt


def remap(h_re, h_im):
    """
    Map (h_re, h_im) to (A, phi)

    The convention is PyART's usual one, h = A exp(-i phi): for a complex
    z = h_re + i h_im, this returns the same phase as
    get_multipole_dict(z)["p"], since unwrap(angle(conj(z))) == -unwrap(angle(z)).
    For polarizations (h = hp - i hc) call it as remap(hp, -hc).

    Parameters
    ----------
    h_re: ndarray
        real part of the waveform
    h_im: ndarray
        imaginary part of the waveform
    Returns
    -------
    out: (A, phi)
        amplitude and phase of the waveform
    """
    amp = np.sqrt(h_re**2 + h_im**2)
    phase = np.unwrap(np.angle(h_re - 1j * h_im))
    return amp, phase


def shift_waveform(h_re, h_im, t_shift_idx, phi_shift):
    """
    Shift waveform in time and phase
    Parameters
    ----------
    h_re: ndarray
        real part of the waveform
    h_im: ndarray
        imaginary part of the waveform
    t_shift_idx: int
        time shift in number of indices
    phi_shift: float
        phase shift in radians
    Returns
    -------
    out: (h_re, h_im)
        shifted real and imaginary parts of the waveform
    """
    h_re = np.roll(h_re, -t_shift_idx)
    h_im = np.roll(h_im, -t_shift_idx)
    A, phi = remap(h_re, h_im)
    phi = phi - phi_shift
    return A * np.cos(phi), -A * np.sin(phi)


#####################################################
#   SimInspiralFD-style conditioning (needs LAL)    #
#####################################################
# These reproduce what LALSimulation's SimInspiralFD does to a time-domain
# waveform before Fourier transforming it: taper the start, high-pass at f_min,
# and taper the filter transients at both ends. They live here so that anything
# generating a frequency-domain waveform from time-domain data -- the bilby NR
# source models, and in principle the match code -- can share them.
#
# Much of this is adapted from pyseobnr:
#   https://github.com/AEI-ACR/pyseobnr/blob/70e730002061210264dcc297d01e6b4c8c5a0bc1/pyseobnr/generate_waveform.py#L1258
#
# Note this is a different thing from analysis.match.condition_td_waveform,
# which pads and Tukey-tapers a pycbc TimeSeries around the merger for mismatch
# computations. The two are deliberately kept separate.
#
# lal/lalsimulation are imported inside the functions: PyART.waveform imports
# this module, and `import PyART` should not require LAL to be installed.


def compute_conditioning_parameters(
    mass_1,
    mass_2,
    spin_1z,
    spin_2z,
    minimum_frequency,
    extra_time_fraction=0.1,
    extra_cycles=3.0,
):
    """
    Time and frequency bounds used by the SimInspiralFD conditioning.

    Parameters
    ----------
    mass_1, mass_2: float
        component masses in Solar masses
    spin_1z, spin_2z: float
        dimensionless spin components along the orbital angular momentum
    minimum_frequency: float
        requested lower frequency of the analysis, in Hz
    extra_time_fraction: float
        fraction of the chirp time used for the Stage-1 taper
    extra_cycles: float
        number of cycles at f_lower prepended before the waveform

    Returns
    -------
    f_start : float
        lower starting frequency for the TD waveform call (includes extra time)
    f_lower : float
        effective minimum frequency used for the chirp-time bound; equal to
        min(minimum_frequency, ISCO_9M), following the SimInspiralFD convention
    f_isco : float
        Schwarzschild ISCO frequency, used for Stage 2
    chirp_time : float
        chirp time at f_lower
    extra_time : float
        extra time (extra_cycles / f_lower) prepended before the waveform
    extra_time_fraction : float
        echoed back, so callers can rebuild the taper length
    """
    import lal
    import lalsimulation

    # SimInspiralFD uses r = 9M ISCO to decide whether to lower f_min.
    fico_check = 1.0 / (pow(9.0, 1.5) * np.pi * (mass_1 + mass_2) * lal.MTSUN_SI)
    f_lower = min(minimum_frequency, fico_check)

    # Schwarzschild (r = 6M) ISCO for Stage 2.
    f_isco = 1.0 / (pow(6.0, 1.5) * np.pi * (mass_1 + mass_2) * lal.MTSUN_SI)

    chirp_time = lalsimulation.SimInspiralChirpTimeBound(
        f_lower, mass_1 * lal.MSUN_SI, mass_2 * lal.MSUN_SI, spin_1z, spin_2z
    )

    spinkerr = lalsimulation.SimInspiralFinalBlackHoleSpinBound(spin_1z, spin_2z)
    tmerge = lalsimulation.SimInspiralMergeTimeBound(
        mass_1 * lal.MSUN_SI, mass_2 * lal.MSUN_SI
    ) + lalsimulation.SimInspiralRingdownTimeBound(
        (mass_1 + mass_2) * lal.MSUN_SI, spinkerr
    )

    extra_time = extra_cycles / f_lower
    f_start = lalsimulation.SimInspiralChirpStartFrequencyBound(
        (1.0 + extra_time_fraction) * chirp_time + tmerge + extra_time,
        mass_1 * lal.MSUN_SI,
        mass_2 * lal.MSUN_SI,
    )

    return f_start, f_lower, f_isco, chirp_time, extra_time, extra_time_fraction


def condition_td_polarisations(
    hp,
    hc,
    delta_t,
    epoch,
    minimum_frequency,
    chirp_time,
    extra_time,
    f_isco,
    extra_time_fraction=0.1,
):
    """
    Apply the SimInspiralFD conditioning to time-domain polarizations.

    Stage 1 tapers the initial extra segment before f_min, high-passes at f_min
    and removes the zero padding. Stage 2 tapers one cycle at f_min from the
    start and one at f_ISCO from the end, killing the filter transients.
    https://github.com/lscsoft/lalsuite/blob/06047f8491f5cf480c720b47e2ad7bf85b26adc4/lalsimulation/lib/LALSimInspiral.c#L5504

    Parameters
    ----------
    hp, hc: ndarray
        time-domain polarizations, uniformly sampled at delta_t
    delta_t: float
        sampling interval in seconds
    epoch: float
        GPS time of the first sample, in seconds
    minimum_frequency: float
        lower frequency of the analysis, in Hz
    chirp_time, extra_time, f_isco, extra_time_fraction:
        as returned by compute_conditioning_parameters

    Returns
    -------
    out: (hp_ts, hc_ts)
        conditioned LAL REAL8TimeSeries

    Raises
    ------
    ValueError
        if the record is shorter than the Stage-1 taper. SimInspiralFD generates
        from a frequency low enough that the taper has somewhere to act; a
        pre-existing record (e.g. an NR simulation) is however long it is, and
        SimInspiralTDConditionStage1 aborts the process rather than returning an
        error, so the check has to happen here.
    """
    import lal
    import lalsimulation

    taper_length = extra_time_fraction * chirp_time + extra_time
    record_length = len(hp) * delta_t
    if record_length <= taper_length:
        raise ValueError(
            f"the record is {record_length:.3f} s long, but the SimInspiralFD "
            f"conditioning wants to taper {taper_length:.3f} s of it at "
            f"minimum_frequency = {minimum_frequency:.1f} Hz. Raise the total "
            f"mass, raise minimum_frequency, or use a longer waveform."
        )

    lal_epoch = lal.LIGOTimeGPS(float(epoch))
    length = len(hp)
    hp_ts = lal.CreateREAL8TimeSeries(
        "hplus", lal_epoch, 0, delta_t, lal.DimensionlessUnit, length
    )
    hc_ts = lal.CreateREAL8TimeSeries(
        "hcross", lal_epoch, 0, delta_t, lal.DimensionlessUnit, length
    )
    hp_ts.data.data[:] = hp
    hc_ts.data.data[:] = hc

    lalsimulation.SimInspiralTDConditionStage1(
        hp_ts, hc_ts, taper_length, minimum_frequency
    )
    lalsimulation.SimInspiralTDConditionStage2(hp_ts, hc_ts, minimum_frequency, f_isco)
    return hp_ts, hc_ts


def fd_polarisations_from_td(
    hp_ts, hc_ts, sampling_frequency, delta_f, maximum_frequency=None
):
    """
    Resize conditioned TD polarizations to a power-of-2 chirp length and FFT
    them with LAL's REAL8TimeFreqFFT.

    Adapted from:
    https://github.com/AEI-ACR/pyseobnr/blob/70e730002061210264dcc297d01e6b4c8c5a0bc1/pyseobnr/generate_waveform.py#L1333

    Parameters
    ----------
    hp_ts, hc_ts: lal.REAL8TimeSeries
        conditioned time-domain polarizations
    sampling_frequency: float
        sampling rate in Hz
    delta_f: float
        frequency spacing of the output; 0 means "use the natural resolution"
    maximum_frequency: float or None
        upper frequency of the analysis, in Hz (default: Nyquist)

    Returns
    -------
    frequency_array : ndarray
        one-sided frequencies [0 .. Nyquist]
    hptilde, hctilde : lal.COMPLEX16FrequencySeries
        one-sided frequency-domain polarizations
    """
    import lal

    nyquist = sampling_frequency // 2
    if maximum_frequency is None:
        maximum_frequency = nyquist

    if delta_f != 0:
        n = int(np.round(maximum_frequency / delta_f))
        if n & (n - 1):
            exp = np.frexp(n)
            nyquist = np.ldexp(1, int(exp[1])) * delta_f

    delta_t = 0.5 / nyquist

    if delta_f == 0:
        chirp_length = hp_ts.data.length
        exp = np.frexp(chirp_length)
        chirp_length = int(np.ldexp(1, int(exp[1])))
        delta_f = 1.0 / (chirp_length * delta_t)
    else:
        chirp_length = int(1.0 / (delta_f * delta_t))

    # Keep the last chirp_length samples (i.e. keep the merger / ringdown).
    # When the waveform is shorter than the segment the first argument is
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

    frequency_array = np.arange(len(hptilde.data.data)) * hptilde.deltaF
    return frequency_array, hptilde, hctilde
