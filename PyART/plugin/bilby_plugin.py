"""
Bilby frequency-domain source models backed by PyART's NR catalogues.

The physics lives in the Waveform class and in PyART.utils.wf_utils; what is
here is catalogue loading, caching, and the bilby calling convention.

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

import numpy as np

from ..utils import utils as ut
from ..utils import wf_utils as wf_ut

logger = logging.getLogger(__name__)


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

    The returned object is the cached one and is shared by every caller, so it
    must not be modified: use ``Waveform.copy()`` before doing anything in place.
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

        def load():
            from ..catalogs import sxs as sxs_catalog

            return sxs_catalog.Waveform_SXS(
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

    elif catalog == "gra":
        ext = waveform_arguments.get("ext", "ext")
        res = waveform_arguments.get("res", "128")
        r_ext = waveform_arguments.get("r_ext", None)
        key = ("gra", ID, path, ellmax, cut_U, load_m0, ext, res, r_ext)

        def load():
            from ..catalogs import gra as gra_catalog

            return gra_catalog.Waveform_GRA(
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
        raise NotImplementedError(
            f"catalog '{catalog}' is not implemented; supported: 'sxs', 'gra'"
        )

    if key not in _WAVEFORM_CACHE:
        logger.info("loading %s simulation %s from %s", catalog.upper(), ID, path)
        _WAVEFORM_CACHE[key] = load()
    return _WAVEFORM_CACHE[key]


def clear_waveform_cache():
    """
    Drop every memoised simulation.

    The cache is unbounded and each entry holds the full mode data, so this is
    worth calling between runs over many simulations. It is also what the tests
    use to make each case independent.
    """
    _WAVEFORM_CACHE.clear()


def component_masses_and_spins(metadata, total_mass):
    """
    Component masses and spins in LAL's mass ordering (mass_1 >= mass_2).

    NR metadata labels the two bodies however the simulation did; LAL requires
    the heavier one to be body 1. Exchanging the labels flips the x axis, so it
    is not enough to swap: the in-plane spin components are swapped *and*
    negated, while the z components are swapped as they are. Together these
    amount to a rigid rotation of the system by pi about z, which the caller
    undoes by flipping the sign of the odd-m modes (see
    nr_time_domain_polarisations).

    Ported from PrecessingNRSur_switch_labels_if_needed in LALSimulation:
    https://github.com/lscsoft/lalsuite/blob/8b202b25185d553f268c64312095da72539614b0/lalsimulation/lib/LALSimIMRPrecessingNRSur.c#L2059

    Parameters
    ----------
    metadata: dict
        PyART metadata, providing m1, m2 and chi{1,2}{x,y,z}
    total_mass: float
        total mass of the binary in Solar masses

    Returns
    -------
    out: (mass_1, mass_2, spin_1, spin_2, labels_switched)
        masses in Solar masses, spins as (x, y, z) dimensionless vectors, and
        whether the two bodies were relabelled
    """
    m1, m2 = float(metadata["m1"]), float(metadata["m2"])
    mass_1 = total_mass * m1 / (m1 + m2)
    mass_2 = total_mass * m2 / (m1 + m2)
    spin_1 = np.array([float(metadata[f"chi1{c}"]) for c in "xyz"])
    spin_2 = np.array([float(metadata[f"chi2{c}"]) for c in "xyz"])

    labels_switched = mass_1 < mass_2
    if labels_switched:
        mass_1, mass_2 = mass_2, mass_1
        # in-plane components: swap and negate; z component: swap only
        flip = np.array([-1.0, -1.0, 1.0])
        spin_1, spin_2 = spin_2 * flip, spin_1 * flip

    return mass_1, mass_2, spin_1, spin_2, labels_switched


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
    catalog = str(waveform_arguments["catalog"]).lower()

    # The cached Waveform is shared, and everything below works in place.
    waveform = load_nr_waveform(**waveform_arguments).copy()
    mode_array = resolve_mode_array(waveform, waveform_arguments)
    *_, labels_switched = component_masses_and_spins(waveform.metadata, total_mass)

    # t = 0 goes on the peak of the frame-invariant amplitude, which is where
    # LAL puts it.
    peak_u = waveform.find_max(
        kind="argmax", modes=mode_array, refine=True, wave="hlm"
    )[0]

    # Drop the tail of the record if asked: GRAthena++ ringdowns drift at late
    # times. t_final is an absolute cut in the catalogue's own retarded time u,
    # t_after_peak is anchored to the merger instead, so it survives a change of
    # cut_U or resolution. If both are given, the earlier one wins.
    t_after_peak = waveform_arguments.get(
        "t_after_peak", _DEFAULT_T_AFTER_PEAK.get(catalog)
    )
    t_final = waveform_arguments.get("t_final", None)
    bounds = [
        bound
        for bound in (t_final, None if t_after_peak is None else peak_u + t_after_peak)
        if bound is not None
    ]
    if bounds:
        u_max = min(bounds)
        if u_max < waveform.u[-1]:
            logger.debug(
                "trimming NR data at u <= %.1f M (was %.1f M)", u_max, waveform.u[-1]
            )
            waveform.cut(waveform.u[-1] - u_max, from_the_end=True, cut_dothlm=True)

    peak_t = peak_u * total_mass * ut.consts["Msun"]
    waveform.to_SI(total_mass, luminosity_distance)

    delta_t = 1.0 / sampling_frequency
    time = np.arange(waveform.u[0] - peak_t, waveform.u[-1] - peak_t, delta_t)
    _, hlm = waveform.interpolate_hlm(
        new_u=time + peak_t, kind="cubic", modes=mode_array
    )

    if labels_switched:
        # Relabelling the bodies rotated the system by pi about z. Undo it by
        # flipping the odd-m modes; the even-m ones are insensitive to it.
        # LALSimIMRPrecessingNRSur.c:2283
        hlm = {
            (ell, emm): wf_ut.get_multipole_dict(((-1) ** emm) * mode["z"])
            for (ell, emm), mode in hlm.items()
        }

    # assume_symmetry=False: the NR catalogues load the m<0 modes independently
    # rather than deriving them, and the simulation need not be non-precessing.
    hplus, hcross = wf_ut.compute_hphc(
        hlm, phi=phase, i=iota, modes=mode_array, assume_symmetry=False
    )
    return time, hplus, hcross


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
    mass_1, mass_2, spin_1, spin_2, _ = component_masses_and_spins(
        waveform.metadata, total_mass
    )

    _, _, f_isco, chirp_time, extra_time, extra_time_fraction = (
        wf_ut.compute_conditioning_parameters(
            mass_1, mass_2, spin_1[2], spin_2[2], minimum_frequency
        )
    )

    time, hplus, hcross = nr_time_domain_polarisations(
        total_mass, luminosity_distance, iota, phase, **waveform_arguments
    )

    # Same treatment before Fourier transforming as SimInspiralFD applies.
    hp_ts, hc_ts = wf_ut.condition_td_polarisations(
        hplus,
        hcross,
        1.0 / sampling_frequency,
        float(time[0]),
        minimum_frequency,
        chirp_time,
        extra_time,
        f_isco,
        extra_time_fraction=extra_time_fraction,
    )
    _, hptilde, hctilde = wf_ut.fd_polarisations_from_td(
        hp_ts, hc_ts, sampling_frequency, delta_f, maximum_frequency=maximum_frequency
    )

    hplus_output = np.zeros_like(frequency_array, dtype=complex)
    hcross_output = np.zeros_like(frequency_array, dtype=complex)
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
    time_shift = np.exp(-2j * np.pi * frequency_array[frequency_bounds] * dt)
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
    """
    :func:`nr_frequency_domain_source_model` pinned to the SXS catalogue.

    Written out as a plain ``def`` rather than a ``functools.partial`` because
    bilby resolves ``frequency-domain-source-model`` by dotted name from an ini
    file and then introspects the signature to work out which sampled
    parameters to pass; a partial object does not expose one. Having an entry
    point per catalogue also means an ini file cannot typo the ``catalog``
    waveform-argument, since it does not have to supply it.
    """
    return nr_frequency_domain_source_model(
        frequency_array,
        total_mass,
        luminosity_distance,
        iota,
        phase,
        **{**waveform_arguments, "catalog": "sxs"},
    )


def gra_frequency_domain_source_model(
    frequency_array,
    total_mass,
    luminosity_distance,
    iota,
    phase,
    **waveform_arguments,
):
    """
    :func:`nr_frequency_domain_source_model` pinned to the GRAthena++ catalogue.

    See :func:`sxs_frequency_domain_source_model` for why this is a plain
    function rather than a partial.
    """
    return nr_frequency_domain_source_model(
        frequency_array,
        total_mass,
        luminosity_distance,
        iota,
        phase,
        **{**waveform_arguments, "catalog": "gra"},
    )
