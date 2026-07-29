import os, json, random, time, copy, tempfile
import numpy as np
from scipy import optimize
from scipy.signal import hilbert
from scipy.stats import norm
import logging

from .match import Matcher
from ..models import teob, seob
from ..models.teob import PotentialMinimum
from ..utils import utils as ut


def log_likelihood_from_mismatch(mm, settings):
    """Turn a mismatch into a log-likelihood for Bayesian sampling.

    `settings` (typically Optimizer.likelihood_settings) selects the form via
    settings["kind"]:

      linear (default): logL = -rho^2 * mm
          Leading-order matched-filter result once time/phase are maximized
          over (which Matcher already does). One physically interpretable
          knob (rho, a reference SNR). Linear in mm, so unlike the Gaussian
          forms below it cannot collapse to an artificially narrow spike.

      gauss_nr: logL = -0.5*(mm/sigma)^2, sigma = settings["mm_nr"]
          Directly encodes "differences below the NR resolution floor are
          meaningless". CAUTION: if the irreducible model mismatch is >>
          mm_nr, this collapses the posterior to a fictitiously narrow spike
          around the minimum -- compare mm_opt to mm_nr before trusting this
          option.

      distinguish: sigma = D / (2*rho^2)  [Lindblom-Owen-Brown]
          A fixed, NR-independent distinguishability scale. D is the
          dimension of the *intrinsic waveform parameter space*, not the
          number of parameters currently being calibrated; pass it
          explicitly via settings["D"], never inferred from kys.

      quadrature: sigma^2 = mm_nr^2 + (D/(2*rho^2))^2
          Combines the two Gaussian scales above; never claims precision
          better than either the NR resolution or the detector's ability to
          distinguish waveforms. Robust to gauss_nr's collapse mode and to
          an anomalously small mm_nr from a badly-converged Lev pair.

    Returns -inf for missing/non-finite/negative mm (e.g. a failed EOB
    generation should be mapped to -inf by the caller before ever reaching
    here -- see Optimizer._sampler_log_likelihood).
    """
    if mm is None or not np.isfinite(mm) or mm < 0:
        return -np.inf

    kind = settings.get("kind", "linear")
    rho = settings.get("rho", 30.0)

    if kind == "linear":
        return -(rho**2) * mm

    if kind == "gauss_nr":
        sigma = settings.get("mm_nr")
        if not sigma or sigma <= 0:
            raise ValueError(
                "likelihood kind='gauss_nr' requires a positive settings['mm_nr']"
            )
        return -0.5 * (mm / sigma) ** 2

    if kind == "distinguish":
        D = settings.get("D")
        if D is None:
            raise ValueError("likelihood kind='distinguish' requires settings['D']")
        sigma = D / (2.0 * rho**2)
        return -0.5 * (mm / sigma) ** 2

    if kind == "quadrature":
        D = settings.get("D")
        mm_nr = settings.get("mm_nr")
        if D is None or not mm_nr:
            raise ValueError(
                "likelihood kind='quadrature' requires settings['D'] and settings['mm_nr']"
            )
        sigma = np.sqrt(mm_nr**2 + (D / (2.0 * rho**2)) ** 2)
        return -0.5 * (mm / sigma) ** 2

    raise ValueError(f"Unknown likelihood kind: {kind!r}")


class Optimizer(object):
    """
    Class to compute EOB initial data that minimize mismatch
    with reference waveform.
    """

    def __init__(
        self,
        ref_Waveform,
        model='teob',
        kind_ic="E0pph0",
        vrs=[
            "H_hyp",
            "j_hyp",
        ],
        map_function=None,
        use_nqc=True,
        r0_eob=None,
        model_opts={},
        opt_max_iter=1,
        opt_good_mm=5e-3,
        opt_bounds=None,
        bounds_iter={},
        minimizer={"kind": "dual_annealing"},
        use_matcher_cache=False,
        json_file=None,
        overwrite=False,
        json_save_dyn=False,
        mm_settings=None,
        objective_settings=None,
        likelihood_settings=None,
        verbose=True,
        debug=False,
    ):
        """
        Initialize the optimizer for initial conditions (ICs) of EOB waveform models.
        Parameters
        ----------
        ref_Waveform : Waveform
            Reference waveform object containing metadata and data to optimize against.
        kind_ic : str, optional
            Kind of initial conditions to optimize over (default: "E0pph0").
        vrs : list of str, optional
            Variables to optimize over (default: ["H_hyp", "j_hyp"]).
        map_function : callable, optional
            Function to map variables to EOB parameters. If None, uses default mapping.
        use_nqc : bool, optional
            Whether to use NQC corrections in the EOB model (default: True).
        r0_eob : float or None, optional
            Fixed value for r_hyp in EOB model. If None, computed by TEOB (default: None).
        model_opts : dict, optional
            Model-specific options for the EOB waveform generator (default: {}).
        opt_max_iter : int, optional
            Maximum optimization iterations (using different initial guesses) if mismatch threshold is not reached (default: 1).
        opt_good_mm : float, optional
            Interrupt optimization iterations if mismatch is below this threshold (default: 5e-3).
        opt_bounds : dict or None, optional
            Bounds for optimization variables. If None, uses default bounds (default: None).
        bounds_iter : dict, optional
            Options for iterating over bounds during optimization (default: {}).
        minimizer : dict, optional
            Minimizer options, including kind (e.g., "dual_annealing", "dynesty", "differential_evolution", "minimize_scalar") (default: {"kind": "dual_annealing"}).
        use_matcher_cache : bool, optional
            Whether to use cached mismatch computations (default: False).
        json_file : str or None, optional
            Path to JSON file for saving mismatch results. If None, no output is saved (default: None).
        overwrite : bool, optional
            Whether to overwrite existing JSON mismatch data (default: False).
        json_save_dyn : bool, optional
            Whether to save dynamics in the JSON output (default: False).
        mm_settings : dict or None, optional
            Options for the Matcher class (default: None).
        objective_settings : dict or None, optional
            Objective function settings. By default, the optimizer minimizes the
            mismatch at a single reference mass (metric='reference'). It can be
            configured to aggregate mismatch across a mass range with
            metric='mass_range' and aggregate='max' (or mean/median/min).
        verbose : bool, optional
            Whether to print verbose output during optimization (default: True).
        debug : bool, optional
            Whether to enable debug plotting (default: False).
        Notes
        -----
        - Initializes optimizer state, sets up bounds, minimizer, and loads or computes mismatches.
        - Supports multiple optimization strategies and iterative bound expansion.
        - Handles caching and JSON output for mismatch results.
        - Prints warnings for recommended Matcher settings.
        """

        self.ref_Waveform = ref_Waveform
        self.opt_Waveform = None
        self.model = model

        self.kind_ic = kind_ic
        self.use_nqc = use_nqc
        self.r0_eob = r0_eob
        self.model_opts = model_opts

        self.opt_max_iter = opt_max_iter
        self.opt_good_mm = opt_good_mm
        self.opt_data = None
        # Populated by sampler backends (e.g. nessai, dynesty) with anything
        # beyond the (opts, mm_opt) contract -- posterior samples, evidence,
        # etc. -- merged into opt_data after optimize_mismatch's minimize()
        # call. Existing point-estimate backends leave this empty.
        self.sampler_extras = {}
        # Matcher cache for the sampler likelihood path. The point-estimate
        # backends receive the cache through __func_to_minimize's `cache`
        # argument, but sampler backends call _sampler_log_likelihood
        # directly and would otherwise re-condition and re-FFT the (fixed)
        # reference waveform on every single evaluation. Populated by
        # optimize_mismatch under the same guard as that `cache`.
        self._sampler_cache = {}

        self.opt_bounds = opt_bounds

        self.use_matcher_cache = use_matcher_cache

        self.json_file = json_file
        self.json_save_dyn = json_save_dyn
        self.overwrite = overwrite
        self.verbose = verbose
        self.debug = debug

        # mismatch settings
        self.mm_settings = Matcher.__default_parameters__(0)
        if isinstance(mm_settings, dict):
            for k in mm_settings:
                self.mm_settings[k] = mm_settings[k]

        # objective settings
        self.__objective__defaults__()
        if isinstance(objective_settings, dict):
            self.objective_settings = {**self.objective_settings, **objective_settings}

        # likelihood settings (Bayesian sampler backends only)
        self.__likelihood__defaults__()
        if isinstance(likelihood_settings, dict):
            self.likelihood_settings = {
                **self.likelihood_settings,
                **likelihood_settings,
            }
        self._n_failed_likelihood_evals = 0

        if self.mm_settings["cut_longer"] and self.verbose:
            logging.warning(
                "using the option 'cut_longer' during optimization should be avoided!"
            )
        if not self.mm_settings["cut_second_waveform"] and self.verbose:
            logging.warning(
                "using the option 'cut_second_waveform' during optimization is strongly suggested!"
            )

        # decide IC vars based on kind_ic
        self.__set_variables(vrs)
        if map_function is not None:
            if self.map_function is None:
                self.map_function = map_function
            else:
                logging.warning('map_function is not None, but kind_ic is not "choose"')
                logging.warning("         user-input map_function will be ignored.")

        shadowed = set(self.opt_vars) & set(self.model_opts)
        if shadowed:
            raise ValueError(
                f"model_opts contains key(s) also being optimized: {sorted(shadowed)}. "
                "generate_EOB merges 'pars = pars | self.model_opts | model_opts', so "
                "self.model_opts always wins over the trial value produced by "
                "map_function -- the objective would be silently constant in that "
                "variable. Remove it from model_opts, or drop it from vrs/opt_bounds."
            )

        if self.opt_bounds is None:
            self.opt_bounds = {var: [None, None] for var in self.opt_vars}

        # Reference values used to (re)build bounds when they are expanded.
        # For variables not present in metadata, use the center of user-provided
        # bounds when available to keep expansions in a physical region.
        self._bounds_reference = {}
        for ky in self.opt_vars:
            if ky in self.ref_Waveform.metadata:
                self._bounds_reference[ky] = self.ref_Waveform.metadata[ky]
                continue

            ky_bounds = self.opt_bounds.get(ky, [None, None])
            if ky_bounds[0] is not None and ky_bounds[1] is not None:
                self._bounds_reference[ky] = 0.5 * (ky_bounds[0] + ky_bounds[1])
                if self.verbose:
                    logging.warning(
                        f"update bounds, {ky} not found in metadata. "
                        f"Using bound center {self._bounds_reference[ky]:.5f} as reference."
                    )
            else:
                self._bounds_reference[ky] = 1.0
                if self.verbose:
                    logging.warning(
                        f"update bounds, {ky} not found in metadata and no explicit bounds center is available. "
                        "Using 1.0 as reference."
                    )
        # update bounds iterator
        self.__bounds_iter_defaults__()
        self.bounds_iter = {**self.bounds_iter, **bounds_iter}
        # update bounds
        self.__update_bounds(eps=self.bounds_iter["eps_initial"])

        # set minimizer
        self.__minimizer__defaults__()
        self.minimizer = {**self.minimizer, **minimizer}
        self.annealing_counter = 0
        if minimizer["kind"] == "dynesty":
            self.minimize = self.__minimize__dynesty__
        elif minimizer["kind"] == "dual_annealing":
            self.minimize = self.__minimize_annealing_
        elif minimizer["kind"] == "differential_evolution":
            self.minimize = self.__minimize_differential_evo_
        elif minimizer["kind"] == "minimize_scalar":
            self.minimize = self.__minimize_scalar_
        elif minimizer["kind"] == "nessai":
            self.minimize = self.__minimize_nessai__
        elif minimizer["kind"] == "zeus":
            self.minimize = self.__minimize_zeus__
        elif minimizer["kind"] == "pocomc":
            self.minimize = self.__minimize_pocomc__
        elif minimizer["kind"] == "grid":
            self.minimize = self.__minimize_grid__
        elif minimizer["kind"] == "gp_surrogate":
            self.minimize = self.__minimize_gp_surrogate__
        elif minimizer["kind"] == "nautilus":
            self.minimize = self.__minimize_nautilus__
        else:
            raise ValueError(f'Unknown minimizer kind: {minimizer["kind"]}')

        if verbose:
            q = ref_Waveform.metadata["q"]
            chi1 = ref_Waveform.metadata["chi1z"]
            chi2 = ref_Waveform.metadata["chi2z"]
            flags_str = ""
            if "flags" in ref_Waveform.metadata:
                for flag in ref_Waveform.metadata["flags"]:
                    flags_str += flag + ", "
                flags_str = flags_str[:-2]
            logging.info("###########################################")
            logging.info(f"###          Running Optimizer          ###")
            logging.info("###########################################\n")
            logging.info(f'Reference waveform : {ref_Waveform.metadata["name"]}')
            logging.info(f"(q, chi1z, chi2z)  : ({q:.2f}, {chi1:.2f}, {chi2:.2f})")
            logging.info(f"binary type        : {flags_str}")
            logging.info(f"Variables for ICs  : {self.opt_vars}")
            logging.info(f"Objective metric   : {self.objective_settings['metric']}")
            if self.objective_settings["metric"] == "mass_range":
                m0, m1 = self.objective_settings["mass_range"]
                nm = self.objective_settings["num_masses"]
                agg = self.objective_settings["aggregate"]
                logging.info(
                    f"Objective mass span: [{m0:.3f}, {m1:.3f}] with {nm:d} points"
                )
                logging.info(f"Objective aggregate: {agg}")
            logging.info(" ")

        mm_data = self.load_or_create_mismatches()
        ref_name = self.ref_Waveform.metadata["name"]

        run_optimization = True
        opt_data = None
        if ref_name in mm_data["mismatches"]:
            opt_data = mm_data["mismatches"][ref_name]

            if not overwrite or opt_data["mm_opt"] < self.bounds_iter["bad_mm"]:
                run_optimization = False
            if verbose:
                logging.info(f"Loading mismatch from {self.json_file}")
                logging.info("Optimal ICs  :")
                for ky in self.opt_vars:
                    logging.info(
                        f'                {ky:5s} : {opt_data[ky+"_opt"]:.15f}'
                    )
                logging.info("Original mm  : {:.3e}".format(opt_data["mm0"]))
                logging.info("Optimized mm : {:.3e}\n".format(opt_data["mm_opt"]))

        if run_optimization:
            random.seed(self.minimizer["opt_seed"])
            np.random.seed(self.minimizer["opt_seed"])
            dashes = "-" * 45
            asterisks = "*" * 45
            best_waveform = None

            eps = copy.copy(self.bounds_iter["eps_initial"])

            t0 = time.perf_counter()
            # i-loop on different search bounds
            for i in range(1, self.bounds_iter["max_iter"] + 1):
                if self.bounds_iter["max_iter"] > 1 and self.verbose:
                    logging.info(
                        f"\n{asterisks}\nSearch bounds (eps) iteration  #{i:d}\n{asterisks}"
                    )

                # j-loop on different initial gueses
                for j in range(1, self.opt_max_iter + 1):
                    if self.verbose:
                        logging.info(
                            f"{dashes}\nOptimization iteration #{j:d}\n{dashes}"
                        )
                    if (
                        i == 1 and j == 1 and opt_data is None
                    ):  # if first iter of both loops
                        opt_data = self.optimize_mismatch(use_ref_guess=True)
                        best_waveform = self.opt_Waveform
                    else:
                        opt_data_new = self.optimize_mismatch(use_ref_guess=False)
                        candidate_waveform = self.opt_Waveform
                        if opt_data_new["mm_opt"] < opt_data["mm_opt"]:
                            opt_data = opt_data_new
                            best_waveform = candidate_waveform
                        else:
                            # Keep the waveform associated with the global best result.
                            self.opt_Waveform = best_waveform
                    # if we reached a nice mismatch, break loop on initial guesses
                    if opt_data["mm_opt"] <= self.opt_good_mm:
                        break

                if opt_data["mm_opt"] <= self.bounds_iter["bad_mm"]:
                    # if the mismatch is good according to eps-standard, then break
                    break

                elif i < self.bounds_iter["max_iter"]:
                    # otherwise, increase the bound search (if we are not at the last iter)
                    kys = self.opt_vars
                    old_bounds = {ky: list(self.opt_bounds[ky]) for ky in kys}
                    for ky in eps:
                        eps[ky] *= self.bounds_iter["eps_factors"][ky]

                    if self.bounds_iter.get("expand_mode", "legacy") == "monotone":
                        # Re-seat the reference for non-metadata variables onto the
                        # current optimum, so growth is centred on where the search
                        # actually is rather than a value frozen at construction.
                        for ky in kys:
                            opt_key = f"{ky}_opt"
                            if (
                                ky not in self.ref_Waveform.metadata
                                and opt_key in opt_data
                            ):
                                self._bounds_reference[ky] = opt_data[opt_key]

                        candidate_bounds = {}
                        for ky in kys:
                            ref_val = self._bounds_reference[ky]
                            delta = abs(ref_val) * eps[ky]
                            if delta == 0:
                                delta = eps[ky]
                            candidate_bounds[ky] = [ref_val - delta, ref_val + delta]

                        # Union with the previous bounds: expansion never shrinks
                        # the interval the user (or a prior iteration) already had.
                        for ky in kys:
                            self.opt_bounds[ky] = [
                                min(old_bounds[ky][0], candidate_bounds[ky][0]),
                                max(old_bounds[ky][1], candidate_bounds[ky][1]),
                            ]
                    else:
                        self.opt_bounds = {ky: [None, None] for ky in kys}
                        self.__update_bounds(eps=eps)
                    old_bounds_str = ", ".join(
                        [
                            f"{ky}:[{old_bounds[ky][0]:.3f},{old_bounds[ky][1]:.3f}]"
                            for ky in kys
                        ]
                    )
                    new_bounds_str = ", ".join(
                        [
                            f"{ky}:[{self.opt_bounds[ky][0]:.3f},{self.opt_bounds[ky][1]:.3f}]"
                            for ky in kys
                        ]
                    )
                    logging.info(f"\nIncreasing search bounds: {old_bounds_str}")
                    logging.info(f"                  ----> : {new_bounds_str}")

                else:
                    mm_opt = opt_data["mm_opt"]
                    logging.info("\n++++++++++++++++++++++++++++++++++++++")
                    logging.info(
                        f'+++  Reached eps_max_iter : {self.bounds_iter["max_iter"]:2d}     +++'
                    )
                    logging.info(
                        f'+++  mm_opt : {mm_opt:.2e} > {self.bounds_iter["bad_mm"]:.2e}  +++'
                    )
                    logging.info("++++++++++++++++++++++++++++++++++++++")

            mm_data["mismatches"][ref_name] = opt_data

            if verbose:
                logging.info(
                    "\n>> Best mismatch found : {:.3e}".format(opt_data["mm_opt"])
                )
                logging.info(
                    ">> Total elapsed time  : {:.1f} s\n".format(
                        time.perf_counter() - t0
                    )
                )

            if json_file is not None:
                self.save_mismatches(mm_data)
            self.opt_Waveform = best_waveform
        self.opt_data = opt_data
        pass

    def __set_variables(self, vrs):
        """
        Set the variables to optimize over depending on the kind of ICs
        selected.

        Parameters
        ----------
        vrs : list of str
            List of variables to optimize over if kind_ic is "choose".
            Otherwise, set automatically based on kind_ic.

        Raises
        ------
        ValueError
            If kind_ic is unknown.

        Notes
        -----
        - Supported kinds of ICs:
            - "choose": user-defined variables in `vrs`.
            - "e0f0": optimizes over eccentricity `e0` and frequency `f0`.
            - "E0pph0": optimizes over energy `E0byM` and angular momentum `pph0`.
            - "phi0theta0": optimizes over in-plane spin rotation angle `theta` and reference phase `phi_ref`.
        """
        if self.kind_ic == "choose":
            self.opt_vars = vrs
            self.map_function = None  # Needs to be defined by the user
        elif self.kind_ic == "e0f0":
            self.opt_vars = ["e0", "f0"]
            self.map_function = lambda x: {
                "ecc": x["e0"],
                "f0": x["f0"],
            }  # map to EOB pars
        elif self.kind_ic == "E0pph0":
            self.opt_vars = ["E0byM", "pph0"]
            self.map_function = lambda x: {"H_hyp": x["E0byM"], "J_hyp": x["pph0"]}
        elif self.kind_ic == "phi0theta0":
            self.opt_vars = ["phi_ref", "theta"]

            def rotate_in_plane_spins(chiA, chiB, theta=0.0):
                """
                Perform a rotation of the in-plane spins by an angle theta
                """
                from scipy.spatial.transform import Rotation

                zaxis = np.array([0, 0, 1])
                r = Rotation.from_rotvec(theta * zaxis)
                chiA_rot = r.apply(chiA)
                chiB_rot = r.apply(chiB)
                return chiA_rot, chiB_rot

            def func(vrs):
                theta = vrs["theta"]
                phi_ref = vrs["phi_ref"]
                chiA = np.array([vrs["chi1x"], vrs["chi1y"], vrs["chi1z"]])
                chiB = np.array([vrs["chi2x"], vrs["chi2y"], vrs["chi2z"]])

                # rotate in-plane spin components by theta
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, theta=theta)
                rotated = {
                    "chi1x": chiA_rot[0],
                    "chi1y": chiA_rot[1],
                    "chi1z": chiA_rot[2],
                    "chi2x": chiB_rot[0],
                    "chi2y": chiB_rot[1],
                    "chi2z": chiB_rot[2],
                    "phi_ref": phi_ref,
                }
                return rotated

            self.map_function = func
        else:
            raise ValueError(f"Unknown kind of ICs: {self.kind_ic}")
        pass

    def __update_bounds(self, eps=None):
        """
        Set the bounds for the optimization; if the bounds are not specified,
        set them to the reference value (read from metadata) +/- eps

        Parameters
        ----------
        eps : dict, optional
            Dictionary with the epsilon values for each variable.
            If None, use self.bounds_iter["eps_initial"].

        Notes
        -----
        - If a bound is already specified (not None), it is not updated.
        - Epsilon values define the relative range around the reference value.
        - If a variable is not found in the reference metadata, a warning is printed and its value is set to 1.
        """
        if eps is None:
            eps = self.bounds_iter["eps_initial"]
        default_bounds = {}
        for ky in self.opt_vars:
            ref_val = self._bounds_reference[ky]
            delta = abs(ref_val) * eps[ky]
            if delta == 0:
                delta = eps[ky]
            default_bounds[ky] = [ref_val - delta, ref_val + delta]

        for ky in self.opt_vars:
            for j in range(2):
                if self.opt_bounds[ky][j] is None:
                    self.opt_bounds[ky][j] = default_bounds[ky][j]
        pass

    def load_or_create_mismatches(self):
        """
        Load mismatches data if the options of the
        json file stored is consistent with current ones.
        Otherwise, create a new dictionary (NOT a new json file)

        Returns
        -------
        data : dict
            Dictionary containing the options and mismatches data.
        Raises
        ------
        RuntimeError
            If the options in the JSON file differ from the current ones.
        Notes
        -----
        - Compares current options with those in the JSON file to ensure consistency.
        - If the JSON file does not exist, initializes a new data structure.
        """
        # convert numpy array to lists to avoid issues with JSON writing/loading
        loc_mm_settings = copy.deepcopy(self.mm_settings)
        for k in loc_mm_settings:
            val = loc_mm_settings[k]
            if isinstance(val, np.ndarray):
                loc_mm_settings[k] = list(val)
        del loc_mm_settings["initial_frequency_mm"]  # save this at sim-level
        del loc_mm_settings["final_frequency_mm"]

        # options to store/read in JSON
        options = {
            "minimizer": self.minimizer,
            "kind_ic": self.kind_ic,
            "vars": self.opt_vars,
            "mm_settings": loc_mm_settings,
        }

        # check if file exists
        if self.json_file is not None and os.path.exists(self.json_file):
            # load mismatches dict
            with open(self.json_file, "r") as file:
                json_data = json.loads(file.read())

            # fix list of list to list of tuples for 'modes' in json-data
            modes_list_of_list = json_data["options"]["mm_settings"]["modes"]
            json_data["options"]["mm_settings"]["modes"] = [
                tuple(mode) for mode in modes_list_of_list
            ]

            # check that the options are the same:
            # 1) start by checking everything except mm_settings
            # 2) then check mm_settings
            dicts2check = [
                [json_data["options"], options],
                [json_data["options"]["mm_settings"], options["mm_settings"]],
            ]
            names = [["json", "self"], ["mm_set-json", "mm_set-self"]]
            list_excluded_keys = [
                ["mm_settings"],
                ["debug", "initial_frequency_mm", "final_frequency_mm"],
            ]
            for i in range(len(dicts2check)):
                dict1 = dicts2check[i][0]
                dict2 = dicts2check[i][1]
                name1 = names[i][0]
                name2 = names[i][1]
                excl_keys = list_excluded_keys[i]
                if not ut.are_dictionaries_equal(
                    dict1, dict2, excluded_keys=excl_keys, verbose=True
                ):
                    ut.print_dict_comparison(
                        dict1,
                        dict2,
                        excluded_keys=excl_keys,
                        dict1_name=name1,
                        dict2_name=name2,
                    )
                    raise RuntimeError(
                        "The options in the json-file are different w.r.t. the currient ones. Exit."
                    )

            data = json_data
        else:
            # create mismatches dict
            data = {"options": options, "mismatches": {}}
        return data

    def save_mismatches(self, data, verbose=None, json_file=None, overwrite=None):
        """
        Save the mismatches data to a JSON file.

        Parameters
        ----------
        data : dict
            Dictionary containing the options and mismatches data.
        verbose : bool, optional
            Whether to print verbose output (default: self.verbose).
        json_file : str or None, optional
            Path to JSON file for saving mismatch results (default: self.json_file).
        overwrite : bool, optional
            Whether to overwrite existing JSON mismatch data (default: self.overwrite).

        Notes
        -----
        - If the JSON file already exists and contains data for the current simulation,
            it will not be overwritten unless `overwrite` is set to True.
        - If the JSON file does not exist, it will be created.
        """

        if verbose is None:
            verbose = self.verbose
        if overwrite is None:
            overwrite = self.overwrite
        if json_file is None:
            json_file = self.json_file
        if json_file is None:  # i.e., if self.json_file is None
            pass

        sim_name = self.ref_Waveform.metadata["name"]
        creating_new_file = True
        if os.path.exists(json_file) and not overwrite:
            with open(json_file, "r") as file:
                json_data = json.loads(file.read())

            creating_new_file = False
            if sim_name in json_data["mismatches"]:
                logging.info(
                    f"   ---> File {json_file} alreay exists and contains {sim_name}, but overwriting is off."
                )
                json_file = json_file.replace(".json", "_new.json")
                logging.info(f"   ---> writing on file: {json_file}")
                creating_new_file = True

        with open(json_file, "w") as file:
            file.write(json.dumps(data, indent=2))

        if verbose:
            action = "Created" if creating_new_file else "Updated"
            logging.info(f"{action} {json_file}\n")
        pass

    def generate_EOB(self, model="teob", ICs={"f0": None, "e0": None}, model_opts={}):
        """
        Generate an EOB waveform with given initial conditions (ICs).
        TODO: generalise this to any model

        Parameters
        ----------
        ICs : dict, optional
            Dictionary containing the initial conditions to set in the EOB model.
            The keys depend on the kind of ICs selected (default: {"f0": None, "e0": None}).

        Returns
        -------
        eob_wave : Waveform_EOB or None
            Generated EOB waveform object, or None if generation failed.
        model_opts : dict
           Additional options to use for EOB generation
        Notes
        -----
        - Maps the provided ICs to EOB parameters using the specified mapping function.
        - Sets additional intrinsic parameters from the reference waveform metadata.
        - Handles special cases for certain ICs (e.g., "H_hyp", "J_hyp") and r0_eob.
        - Catches exceptions during EOB waveform generation and returns None if an error occurs.
        """

        ref_meta = self.ref_Waveform.metadata
        # Set all the intrinsic parameters that are not in ICs
        default_intrinsic = [
            "M",
            "q",
            "chi1x",
            "chi1y",
            "chi1z",
            "chi2x",
            "chi2y",
            "chi2z",
        ]
        if model == "teob":
            default_intrinsic += [
                "LambdaAl2",
                "LambdaBl2",
            ]
        for ic in ICs:
            if ic in default_intrinsic:
                default_intrinsic.remove(ic)

        if model == "teob":
            if "LambdaAl2" not in ref_meta:
                ref_meta["LambdaAl2"] = 0.0
            if "LambdaBl2" not in ref_meta:
                ref_meta["LambdaBl2"] = 0.0

        sub_meta = {key: ref_meta[key] for key in default_intrinsic}

        # map the ICs (and the other intrinsic pars) to the EOB parameters
        mapped_ids = self.map_function({**ICs, **sub_meta})

        if model == "teob":
            sub_meta["use_nqc"] = self.use_nqc
            sub_meta["ode_tmax"] = 3e5
            if "H_hyp" in mapped_ids or "J_hyp" in mapped_ids:
                if self.r0_eob == "read":
                    # start close to the NR value, a little earlier
                    mapped_ids["r_hyp"] = ref_meta["r0"] * 1.1
                else:
                    if self.r0_eob is not None:
                        if self.r0_eob < ref_meta["r0"]:
                            logging.warning(
                                f'r0_eob={self.r0_eob} is smaller than the NR value r0={ref_meta["r0"]}'
                            )
                            logging.warning("         Setting r0_eob to NR value")
                            mapped_ids["r_hyp"] = ref_meta["r0"]
                        else:
                            mapped_ids["r_hyp"] = self.r0_eob
                    mapped_ids["r_hyp"] = (
                        self.r0_eob
                    )  # if None, it will be computed in the EOB model

        # add the mapped ICs to the sub_meta dictionary & additional model options
        # and run
        sub_meta.update(mapped_ids)
        try:
            if model == "teob":
                dict_func = teob.CreateDict
                eob_model = teob.Waveform_EOB
            elif model == "seob":
                dict_func = seob.CreateDict
                eob_model = seob.Waveform_SEOB
            else:
                raise ValueError(f"Unknown model: {model}")
            pars = dict_func(**sub_meta)
            pars = pars | self.model_opts | model_opts
            eob_wave = eob_model(pars=pars)
        except Exception as e:
            logging.warning(f"Error occurred in EOB wave generation:\n{e}")
            eob_wave = None
        return eob_wave

    def __objective__defaults__(self):
        """Set default options for the optimization objective."""
        ref_mass = self.mm_settings.get("M", self.ref_Waveform.metadata.get("M", 1.0))
        self.objective_settings = {
            "metric": "reference",  # reference | mass_range
            "mass_range": [float(ref_mass), float(ref_mass)],
            "num_masses": 11,
            "aggregate": "max",  # max | mean | median | min
            "rescale_initial_frequency": True,
        }

    def __likelihood__defaults__(self):
        """Set default likelihood options (Bayesian sampler backends only;
        point-estimate backends -- minimize_scalar, dual_annealing,
        differential_evolution -- ignore this entirely)."""
        self.likelihood_settings = {
            "kind": "linear",  # linear | gauss_nr | distinguish | quadrature
            "rho": 30.0,
            "D": None,  # required for distinguish/quadrature; see log_likelihood_from_mismatch
            "mm_nr": None,  # required for gauss_nr/quadrature; the NR error floor
        }

    def _eval_mm_and_logl(self, x, kys):
        """Generate the trial waveform at x and return (mismatch, log-likelihood).

        Used directly by every posterior-producing method (dynesty, nessai,
        pocomc, zeus, grid), rather than through __func_to_minimize, which
        reports a flat mm=1.0 when the ODE integration fails, so a plain
        minimizer still has a well-defined value to descend away from. For a
        posterior that flat value would instead read as ordinary, merely
        low, support -- a point where the integration never produced a
        waveform at all should carry zero probability, not a small one, so
        it is mapped to (nan, -inf) here and tallied in
        self._n_failed_likelihood_evals.
        """
        vs = {kys[i]: x[i] for i in range(len(kys))}
        eob_Waveform = self.generate_EOB(model=self.model, ICs=vs)
        if eob_Waveform is None:
            self._n_failed_likelihood_evals += 1
            return np.nan, -np.inf
        mm = self.objective_mismatch(
            eob_Waveform, verbose=False, iter_loop=False, cache=self._sampler_cache
        )
        logl = log_likelihood_from_mismatch(mm, self.likelihood_settings)
        return mm, logl

    def _sampler_log_likelihood(self, x, kys):
        """Log-likelihood alone, for the samplers that never need the mismatch itself."""
        _, logl = self._eval_mm_and_logl(x, kys)
        return logl

    def _objective_mass_grid(self):
        """Return the mass grid used by the mass-range objective."""
        mass_range = self.objective_settings.get("mass_range", None)
        if not isinstance(mass_range, (list, tuple)) or len(mass_range) != 2:
            raise ValueError(
                "objective_settings['mass_range'] must be a 2-element list/tuple [Mmin, Mmax]"
            )

        m0 = float(mass_range[0])
        m1 = float(mass_range[1])
        if m1 < m0:
            m0, m1 = m1, m0

        num_masses = int(self.objective_settings.get("num_masses", 11))
        if num_masses < 2:
            num_masses = 2

        return np.linspace(m0, m1, num=num_masses)

    def _aggregate_values(self, values, aggregate):
        """Aggregate a 1D array according to the selected rule."""
        vals = np.asarray(values, dtype=float)
        if aggregate == "max":
            return float(np.max(vals))
        if aggregate == "mean":
            return float(np.mean(vals))
        if aggregate == "median":
            return float(np.median(vals))
        if aggregate == "min":
            return float(np.min(vals))
        raise ValueError(
            "objective_settings['aggregate'] must be one of: max, mean, median, min"
        )

    def _log_progress_line(self, message):
        """Log a single-line progress update, ending with carriage return."""
        logger = logging.getLogger()
        handlers = list(logger.handlers)
        if not handlers:
            logging.info(message)
            return

        original_terminators = []
        for handler in handlers:
            if hasattr(handler, "terminator"):
                original_terminators.append((handler, handler.terminator))
                handler.terminator = "\r"

        try:
            logger.info(message)
        finally:
            for handler, terminator in original_terminators:
                handler.terminator = terminator

    def objective_mismatch(
        self, eob_Waveform, verbose=None, iter_loop=False, cache=None
    ):
        """Evaluate the scalar objective used by the optimizer.

        Default behavior is a single mismatch at reference mass (metric='reference').
        Alternative behavior aggregates mismatch over a total-mass grid
        (metric='mass_range').
        """
        if verbose is None:
            verbose = self.verbose
        if cache is None:
            cache = {}
        metric = self.objective_settings.get("metric", "reference")

        ret = None
        if metric == "reference":
            mm_obj = self.match_against_ref(
                eob_Waveform,
                verbose=False,
                iter_loop=False,
                cache=cache,
            )

        if metric == "mass_range":
            if eob_Waveform is None:
                mm_obj = 1.0
            else:
                masses = self._objective_mass_grid()
                mm_values = np.zeros_like(masses)

                ref_mass = float(self.mm_settings.get("M", masses[0]))
                ref_f0_mm = self.mm_settings.get("initial_frequency_mm", None)
                do_rescale_f1 = (
                    bool(self.objective_settings.get("rescale_initial_frequency", True))
                    and ref_f0_mm is not None
                )

                # Do not reuse matcher cache across varying masses.
                for i, mass in enumerate(masses):
                    mm_settings_mass = dict(self.mm_settings)
                    mm_settings_mass["M"] = float(mass)
                    if do_rescale_f1:
                        mm_settings_mass["initial_frequency_mm"] = (
                            float(ref_f0_mm) * ref_mass / float(mass)
                        )
                    mm_values[i] = self.match_against_ref(
                        eob_Waveform,
                        verbose=False,
                        iter_loop=False,
                        cache={},
                        mm_settings=mm_settings_mass,
                    )

                aggregate = self.objective_settings.get("aggregate", "max")
                mm_obj = self._aggregate_values(mm_values, aggregate)

        ret = mm_obj

        if ret is None:
            raise ValueError(
                "objective_settings['metric'] must be one of: reference, mass_range"
            )

        penalty = 0.0
        if self.objective_settings.get("low_freq_align", False):
            # additionally perform a low-frequency alignment
            # with a flat PSD and penalize the result if the
            # merger time difference is > 5 M
            time_factor = self.mm_settings.get("M", 1.0) * ut.consts["Msun"]
            # estimate merger frequency from the inst. NR frequency
            p22 = self.ref_Waveform.hlm[(2, 2)]["p"]
            t22 = self.ref_Waveform.u * time_factor / self.ref_Waveform.metadata["M"]
            omg22 = np.gradient(p22, t22)
            f22 = omg22 / (2 * np.pi)
            imx = np.argmax(self.ref_Waveform.hlm[(2, 2)]["A"])  # merger time index
            f_merger = f22[imx]

            # compute match up to f_merger, but use a flat PSD
            mm_settings_low = dict(self.mm_settings)
            # mm_settings_low["final_frequency_mm"] = f_merger / 2
            mm_settings_low["psd"] = "flat"
            _, matcher = self.match_against_ref(
                eob_Waveform,
                verbose=False,
                iter_loop=False,
                return_matcher=True,
                cache={},
                mm_settings=mm_settings_low,
            )
            # extract the time and phase shifts from the low-frequency match
            out = matcher.match_out
            h1f = out["h1f"]
            h2f = out["h2f"]
            j_shift = out["j_shift"]
            ph_shift = out["ph_shift"]

            # apply the time and phase shifts to the EOB waveform
            h2f_shifted = h2f * np.exp(
                1j * (2 * np.pi * j_shift * h2f.sample_frequencies + ph_shift)
            )
            h1t = h1f.to_timeseries()
            h2t = h2f_shifted.to_timeseries()

            # if we assume that ht is the real part of the complex waveform,
            # we now reconstruct the full complex waveform via Hilbert transform
            h1tc = hilbert(h1t)
            h2tc = hilbert(h2t)

            # find the time of merger (peak of amplitude) in the NR waveform
            mrg_idx1 = np.argmax(np.abs(h1tc))
            mrg_time_1 = h1t.sample_times[mrg_idx1]
            mrg_idx2 = np.argmax(np.abs(h2tc))
            mrg_time_2 = h2t.sample_times[mrg_idx2]
            # transform the time difference into a mismatch penalty (e.g., quadratic)
            # in geom units
            M = self.mm_settings.get("M", 1.0) * ut.consts["Msun"]
            tdiff = mrg_time_1 - mrg_time_2
            penalty = (tdiff / M / 5.0) ** 2
            ret = (mm_obj / 1e-4) ** 2 + penalty

            # import matplotlib.pyplot as plt
            # plt.plot(h1t.sample_times / M, np.abs(h1tc), label="NR")
            # plt.plot(h2t.sample_times / M, np.abs(h2tc), label="EOB")
            # plt.axvline(mrg_time_1 / M, color="k", linestyle="--", label="NR merger")
            # plt.axvline(mrg_time_2 / M, color="r", linestyle="--", label="EOB merger")
            # plt.legend()
            # plt.show()

        if verbose and iter_loop:
            self.annealing_counter += 1
            self._log_progress_line(
                "  >> mismatch - penalty - iter  : {:.3e} - {:.3e} - {:3d}".format(
                    mm_obj, penalty, self.annealing_counter
                )
            )

        return ret

    def match_against_ref(
        self,
        eob_Waveform,
        verbose=None,
        iter_loop=False,
        return_matcher=False,
        cache={},
        mm_settings=None,
    ):
        """
        Compute the mismatch between the reference waveform and the
        provided EOB waveform.
        Parameters
        ----------
        eob_Waveform : Waveform_EOB
            EOB waveform object to compare against the reference.
        verbose : bool, optional
            Whether to print verbose output (default: self.verbose).
        iter_loop : bool, optional
            Whether this is called inside an optimization loop (default: False).
        return_matcher : bool, optional
            Whether to return the Matcher object along with the mismatch (default: False).
        cache : dict, optional
            Cache dictionary for storing intermediate results (default: {}).
        mm_settings : dict or None, optional
            Options for the Matcher class (default: self.mm_settings).
        Returns
        -------
        mm : float
            Computed mismatch value.
        matcher : Matcher or None
            Matcher object if return_matcher is True, otherwise None.
        """
        if verbose is None:
            verbose = self.verbose
        if mm_settings is None:
            mm_settings = self.mm_settings
        if eob_Waveform is not None:
            try:
                matcher = Matcher(
                    self.ref_Waveform, eob_Waveform, settings=mm_settings, cache=cache
                )
                mm = matcher.mismatch
            except Exception as e:
                logging.warning("Error while computing match: ", e)
                matcher = None
                mm = 1.0
        else:
            matcher = None
            mm = 1.0
        if verbose and iter_loop:
            self.annealing_counter += 1
            self._log_progress_line(
                "  >> mismatch - iter  : {:.3e} - {:3d}".format(
                    mm, self.annealing_counter
                )
            )
        if return_matcher:
            return mm, matcher
        else:
            return mm

    def __func_to_minimize(self, x, kys, verbose=None, cache={}):
        if verbose is None:
            verbose = self.verbose
        # reassemble the ICs
        vs = {kys[i]: x[i] for i in range(len(kys))}
        eob_Waveform = self.generate_EOB(model=self.model, ICs=vs)
        if eob_Waveform is not None:
            mm = self.objective_mismatch(
                eob_Waveform, verbose=self.verbose, iter_loop=True, cache=cache
            )
        else:
            if self.kind_ic == "E0pph0":
                pph0 = vs["pph0"]
                ref_meta = self.ref_Waveform.metadata
                q = ref_meta["q"]
                chi1 = ref_meta["chi1z"]
                chi2 = ref_meta["chi2z"]
                rvec = np.linspace(2, 20, num=200)
                Vmin = PotentialMinimum(rvec, pph0, q, chi1, chi2)
                dV = Vmin - vs["E0byM"]
            else:
                dV = 0
            mm = 1 + dV
        return mm

    def optimize_mismatch(self, use_ref_guess=True, verbose=None):
        """
        Optimize the mismatch between the reference waveform and the other model waveform.

        Parameters
        ----------
        use_ref_guess : bool, optional
            Whether to use the reference values as the initial guess for the optimization (default: True).
        verbose : bool, optional
            Whether to print verbose output during optimization (default: self.verbose).
        Returns
        -------
        opt_data : dict
            Dictionary containing the optimization results and metadata.
        """
        if verbose is None:
            verbose = self.verbose
        self.sampler_extras = {}
        self._n_failed_likelihood_evals = 0
        kys = self.opt_vars
        bounds = self.opt_bounds
        meta = self.ref_Waveform.metadata

        # Treat variables which are in common with the reference
        kys_ref = [ky for ky in kys if ky in self.ref_Waveform.metadata]  # common keys
        vs_ref = {
            ky: self.ref_Waveform.metadata[ky] for ky in kys_ref
        }  # reference values
        for ky in kys_ref:
            vv = vs_ref[ky]
            if vv < bounds[ky][0] or vv > bounds[ky][1]:
                logging.warning(
                    "Reference value for {:s} is outside searching interval: {:.2e} not in [{:.2e},{:.2e}]".format(
                        ky, vv, bounds[ky][0], bounds[ky][1]
                    )
                )
        if use_ref_guess:
            # use reference values whenever possible
            vs0 = vs_ref
        else:
            # random initial guess
            vs0 = {ky: np.random.uniform(bounds[ky][0], bounds[ky][1]) for ky in kys}

        # randomly select the variables which are not in common with the reference
        for ky in kys:
            if ky not in kys_ref:
                vs0[ky] = np.random.uniform(bounds[ky][0], bounds[ky][1])

        metric = self.objective_settings.get("metric", "reference")
        eob0 = self.generate_EOB(model=self.model, ICs=vs0)
        if metric == "reference":
            mm0, matcher0 = self.match_against_ref(
                eob0, iter_loop=False, return_matcher=True
            )
        else:
            mm0 = self.objective_mismatch(eob0, iter_loop=False)
            matcher0 = None
        if verbose:
            logging.info(f"Original  mismatch    : {mm0:.3e}")
            logging.info("Optimization interval :")
            for ky in kys:
                logging.info(
                    f"                        {ky:5s} : [{bounds[ky][0]:.3e},{bounds[ky][1]:.3e}]"
                )
            logging.info(f"Initial guess         :")
            for ky in kys:
                logging.info(f"                        {ky:5s} : {vs0[ky]:.15f}")

        if self.use_matcher_cache and metric == "reference":
            if matcher0 is None:
                if verbose:
                    logging.info("+++ First mm-computation failed! Not using cache +++")
                cache = {}
            else:
                cache = {"h1f": matcher0.h1f, "M": matcher0.settings["M"]}
        else:
            cache = {}

        # Same cache, reached by the sampler backends (which bypass
        # __func_to_minimize and call _sampler_log_likelihood directly).
        self._sampler_cache = cache

        # prepare for minimization
        x0 = [vs0[ky] for ky in kys]
        bounds_array = np.array([[bounds[ky][0], bounds[ky][1]] for ky in kys])
        f = lambda x: self.__func_to_minimize(x, kys, verbose=verbose, cache=cache)

        t0_annealing = time.perf_counter()
        opts, mm_opt = self.minimize(f, x0, bounds_array, kys)

        if verbose:
            self._log_progress_line(
                "  >> mismatch - iter  : {:.3e} - {:3d}".format(
                    mm_opt, self.annealing_counter
                )
            )
            logging.info(f"Optimized mismatch    : {mm_opt:.3e}")
            logging.info(f"Optimal ICs           :")
            for ky in kys:
                logging.info(f"                        {ky:5s} : {opts[ky]:.15f}")
            logging.info(
                "Minimization time        : {:.1f} s".format(
                    time.perf_counter() - t0_annealing
                )
            )

        # generate the eob waveform corresponding to the optimal ICs
        eob_opt = self.generate_EOB(model=self.model, ICs=opts)
        self.opt_Waveform = eob_opt

        if self.debug:
            temp_settings = copy.copy(self.mm_settings)
            temp_settings["debug"] = True
            self.match_against_ref(eob_opt, mm_settings=temp_settings)

        opt_data = {
            # store also some attributes, just for convenience
            "M": meta["M"],
            "q": meta["q"],
            "chi1x": meta["chi1x"],
            "chi1y": meta["chi1y"],
            "chi1z": meta["chi1z"],
            "chi2x": meta["chi2x"],
            "chi2y": meta["chi2y"],
            "chi2z": meta["chi2z"],
            "LambdaAl2": meta["LambdaAl2"] if "LambdaAl2" in meta else 0.0,
            "LambdaBl2": meta["LambdaBl2"] if "LambdaBl2" in meta else 0.0,
            "initial_frequency_mm": self.mm_settings["initial_frequency_mm"],
            "final_frequency_mm": self.mm_settings["final_frequency_mm"],
            "objective_settings": self.objective_settings,
            "opt_seed": self.minimizer["opt_seed"],
            "opt_max_iter": self.opt_max_iter,
            "opt_good_mm": self.opt_good_mm,
            "bounds_iter": self.bounds_iter,
            # optimization results
            "bounds": bounds,
            "mm0": mm0,
            "mm_opt": mm_opt,
        }
        for ky in kys:
            opt_data[ky] = vs_ref[ky] if ky in vs_ref else None
            opt_data[ky + "_opt"] = opts[ky]

        if self.sampler_extras:
            # Keep an untouched copy alongside the flattened merge below: a
            # caller wanting everything a given backend measured (not just
            # the fields calibration_core.py's reader already knows to look
            # for) can read this instead of having to name each key twice --
            # once here, once in that reader -- as the set of backends grows.
            opt_data["sampler_extras_raw"] = dict(self.sampler_extras)
            opt_data.update(self.sampler_extras)

        if eob_opt is not None and self.json_save_dyn:
            dyn0 = eob_opt.dyn
            dyn0 = {ky: list(dyn0[ky]) for ky in dyn0.keys()}
        else:
            dyn0 = None
        opt_data["dyn0"] = dyn0

        return opt_data

    def __bounds_iter_defaults__(self):
        """
        Set the default options for the bounds iteration.
        """

        self.bounds_iter = {
            "eps_initial": {},  # initial epsilon values
            "eps_factors": {},  # increase-factor for eps at each eps-iter
            "max_iter": 1,  # If true, iterate on eps-bounds
            "bad_mm": 0.1,  # if after opt_max_iter(s) we are still above this threshold
            "expand_mode": "legacy",  # "legacy" (default, unchanged) or "monotone"
            # legacy: bounds are discarded and rebuilt from scratch each expansion,
            #   from a reference frozen at construction time -- this can *shrink*
            #   the search interval below what the user originally asked for.
            # monotone: bounds only ever grow (union with the previous interval),
            #   and for variables not present in ref_Waveform.metadata the
            #   reference point is re-seated onto the current optimum at each
            #   expansion, so growth is centred on where the search actually is.
        }
        for ky in self.opt_vars:
            self.bounds_iter["eps_initial"][ky] = 1e-2
            self.bounds_iter["eps_factors"][ky] = 2
        pass

    def __minimizer__defaults__(self):
        """
        Set the default minimizer options.
        """
        self.minimizer = {  # annealing options
            "kind": "dual_annealing",
            "opt_maxfun": 1000,
            "opt_seed": 190521,
            "xatol": 1e-4,
            # differiantial_evolution options
            "opt_workers": 1,
            # dynesty options
            "nlive": 10,
            "maxiter": 10000,
            "maxcall": 10000,
            "print_progress": True,
        }
        pass

    def __minimize_annealing_(self, f, x0, bounds_array, kys):
        """
        Minimize with dual annealing.

        Parameters
        ----------
        f : callable
            The objective function to minimize.
        x0 : array-like
            Initial guess for the parameters.
        bounds_array : array-like
            Bounds for each parameter as an array of shape (n, 2).
        kys : list of str
            List of parameter names corresponding to the elements in x0.
        Returns
        -------
        opts : dict
            Dictionary of optimized parameters.
        mm_opt : float
            The minimum value of the objective function found.
        """

        maxiter = self.minimizer.get("opt_maxfun", 1000)
        seed = self.minimizer.get("opt_seed", 190521)
        x0 = x0

        opt_result = optimize.dual_annealing(
            f,
            maxfun=maxiter,
            seed=seed,
            x0=x0,
            bounds=bounds_array,
        )

        opt_pars = opt_result["x"]
        opts = {kys[i]: opt_pars[i] for i in range(len(kys))}
        mm_opt = opt_result["fun"]
        return opts, mm_opt

    def __minimize_differential_evo_(self, f, x0, bounds_array, kys):
        """
        Minimize with differential evolution.

        Parameters
        ----------
        f : callable
            The objective function to minimize.
        x0 : array-like
            Initial guess for the parameters.
        bounds_array : array-like
            Bounds for each parameter as an array of shape (n, 2).
        kys : list of str
            List of parameter names corresponding to the elements in x0.
        Returns
        -------
        opts : dict
            Dictionary of optimized parameters.
        mm_opt : float
            The minimum value of the objective function found.
        """
        maxiter = self.minimizer.get("opt_maxfun", 1000)
        seed = self.minimizer.get("opt_seed", 190521)
        workers = self.minimizer.get("opt_workers", 1)
        x0 = x0

        opt_result = optimize.differential_evolution(
            f,
            maxiter=maxiter,
            seed=seed,
            x0=x0,
            workers=workers,
            bounds=bounds_array,
        )

        opt_pars = opt_result["x"]
        opts = {kys[i]: opt_pars[i] for i in range(len(kys))}
        mm_opt = opt_result["fun"]
        return opts, mm_opt

    def __minimize_scalar_(self, f, x0, bounds_array, kys):
        """
        Minimize with scipy.optimize.minimize_scalar.

        Parameters
        ----------
        f : callable
            The objective function to minimize.
        x0 : array-like
            Initial guess for the parameters. Unused by the bounded scalar solver.
        bounds_array : array-like
            Bounds for each parameter as an array of shape (n, 2).
        kys : list of str
            List of parameter names corresponding to the elements in x0.
        Returns
        -------
        opts : dict
            Dictionary of optimized parameters.
        mm_opt : float
            The minimum value of the objective function found.
        """
        if len(kys) != 1:
            raise ValueError(
                "The 'minimize_scalar' backend can only be used with exactly one optimization variable."
            )

        maxiter = self.minimizer.get("opt_maxfun", 1000)
        xatol = self.minimizer.get("xatol", 1e-5)
        bounds = (bounds_array[0][0], bounds_array[0][1])

        def f_scalar(x):
            return f(np.array([x]))

        opt_result = optimize.minimize_scalar(
            f_scalar,
            bounds=bounds,
            method="bounded",
            options={"maxiter": maxiter, "xatol": xatol},
        )

        opts = {kys[0]: opt_result.x}
        mm_opt = opt_result.fun
        return opts, mm_opt

    def _summarize_posterior(
        self,
        kys,
        samples,
        sampler_name,
        output_dir=None,
        log_evidence=None,
        log_evidence_err=None,
        n_likelihood_evaluations=None,
    ):
        """Build the sampler_extras payload shared by Bayesian backends:
        per-variable summary statistics plus a thinned copy of the posterior
        (<=2000 rows, JSON-safe), with the full chain written to a sidecar
        .npz file when output_dir is given.
        """
        samples = np.asarray(samples)
        extras = {"sampler": sampler_name}
        if log_evidence is not None:
            extras["log_evidence"] = float(log_evidence)
        if log_evidence_err is not None:
            extras["log_evidence_err"] = float(log_evidence_err)
        if n_likelihood_evaluations is not None:
            extras["n_likelihood_evaluations"] = int(n_likelihood_evaluations)

        n = samples.shape[0]
        thin = max(1, n // 2000)
        thinned = samples[::thin]

        param_samples = {}
        for i, ky in enumerate(kys):
            col = samples[:, i]
            param_samples[ky] = thinned[:, i].tolist()
            extras[f"{ky}_median"] = float(np.median(col))
            extras[f"{ky}_mean"] = float(np.mean(col))
            extras[f"{ky}_sigma"] = float(np.std(col))
            for q, label in [(5, "q05"), (16, "q16"), (84, "q84"), (95, "q95")]:
                extras[f"{ky}_{label}"] = float(np.percentile(col, q))
        extras["param_samples"] = param_samples

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            samples_path = os.path.join(output_dir, "posterior.npz")
            np.savez(samples_path, samples=samples, kys=np.array(kys))
            extras["param_samples_file"] = samples_path

        return extras

    def __minimize__dynesty__(self, f, x0, bounds_array, kys):
        """
        Minimize with dynesty.
        NOTE: largely untested!

        Parameters
        ----------
        f : callable
            The objective function to minimize.
        x0 : array-like
            Initial guess for the parameters. UNUSED.
        bounds_array : array-like
            Bounds for each parameter as an array of shape (n, 2).
        kys : list of str
            List of parameter names corresponding to the elements in x0.
        Returns
        -------
        opts : dict
            Dictionary of optimized parameters.
        mm_opt : float
            The minimum value of the objective function found.

        """
        from dynesty import NestedSampler

        # Define the dimensionality of our problem.
        ndim = len(kys)
        progress = self.minimizer.get("print_progress", True)
        nlive = self.minimizer.get("nlive", 1024)
        # Configurable rather than hardcoded, but default stays "rwalk":
        # dynesty's own dimensionality heuristic recommends 'unif' for
        # ndim<10 (cheaper in principle -- no fixed evals-per-point cost the
        # way rwalk's `walks=25` has), but measured head-to-head on this
        # calibration problem, 'unif' was slightly worse at rho=30 (22:14 vs
        # ~16-18min, 2672 vs ~2450 evals) and much worse at rho=100 (stalled,
        # projected far worse than rwalk's 40:37) -- 'unif's rejection
        # efficiency depends on the bounding ellipsoid fitting the
        # constrained region well, which this mm(a6c) surface apparently
        # does not give it. See calibration upgrade plan, Stage 3.E.
        sample = self.minimizer.get("sample", "rwalk")
        bound = self.minimizer.get("bound", "multi")
        # maxiter/maxcall scale off nlive rather than a fixed constant --
        # a hardcoded maxcall=10000 silently truncates a converged-looking
        # but not-actually-converged run once nlive grows past a couple
        # hundred, with no error raised (run_nested just returns early).
        maxiter = self.minimizer.get("maxiter", max(10000, 50 * nlive))
        maxcall = self.minimizer.get("maxcall", max(10000, 200 * nlive))

        def loglike(x):
            """Log-likelihood via log_likelihood_from_mismatch(mm,
            self.likelihood_settings) -- see that function for the
            available forms. Uses _sampler_log_likelihood so a failed EOB
            generation maps to -inf explicitly rather than the flat mm=1.0
            __func_to_minimize (and hence `f`) would otherwise return."""
            logl = self._sampler_log_likelihood(x, kys)
            if np.isnan(logl):
                return -np.inf
            return logl

        def prior_transform(u):
            """
            Map the unit cube to the parameter space, assuming
            uniform priors on the parameters.
            """
            return [
                bounds_array[i][0] + u[i] * (bounds_array[i][1] - bounds_array[i][0])
                for i in range(ndim)
            ]

        # Define our sampler.
        sampler = NestedSampler(
            loglike,
            prior_transform,
            ndim,
            nlive=nlive,
            sample=sample,
            bound=bound,
        )
        sampler.run_nested(
            maxiter=maxiter, maxcall=maxcall, print_progress=progress, dlogz=0.1
        )

        if sampler.results.ncall.sum() >= maxcall or sampler.it >= maxiter:
            logging.warning(
                f"dynesty run_nested stopped by maxcall={maxcall}/maxiter={maxiter}, "
                f"not by reaching dlogz -- the posterior is likely under-converged. "
                f"Raise these (or nlive) if this is unexpected."
            )

        # return just the maxL point
        maxL = sampler.results.logl.argmax()
        opts = {kys[i]: sampler.results.samples[maxL][i] for i in range(len(kys))}
        mm_opt = f(sampler.results.samples[maxL])

        output_dir = self.minimizer.get("output_dir")

        # make the traceplot
        if self.verbose:
            from dynesty import plotting as dyplot

            fig, _ = dyplot.traceplot(
                sampler.results, show_titles=True, trace_cmap="viridis"
            )
            traceplot_path = (
                os.path.join(output_dir, "traceplot.png")
                if output_dir
                else "traceplot.png"
            )
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            fig.savefig(traceplot_path)

        # Nested sampling's raw sample trace is *unequally weighted* -- early
        # samples span the whole prior before the run has converged, and
        # carry negligible posterior mass. Naive statistics over the raw
        # trace are dominated by that early, wide exploration and badly
        # overstate the posterior width. Resample to equal weight first.
        from dynesty.utils import resample_equal

        posterior_samples = resample_equal(
            sampler.results.samples, sampler.results.importance_weights()
        )

        self.sampler_extras = self._summarize_posterior(
            kys,
            posterior_samples,
            "dynesty",
            output_dir=output_dir,
            log_evidence=sampler.results.logz[-1],
            log_evidence_err=sampler.results.logzerr[-1],
            n_likelihood_evaluations=sampler.results.ncall.sum(),
        )
        self.sampler_extras["likelihood_settings"] = dict(self.likelihood_settings)
        self.sampler_extras["n_failed_likelihood_evals"] = (
            self._n_failed_likelihood_evals
        )

        return opts, mm_opt

    def __minimize_nessai__(self, f, x0, bounds_array, kys):
        """Minimize with nessai (nested sampling with normalising-flow
        proposals).

        nessai raises OneDimensionalModelError for single-variable models --
        its flow-based proposals are not designed for 1-D problems. Use the
        'dynesty' backend for single-parameter Bayesian fits (e.g. a6c or
        cN3LO alone); reserve nessai for joint fits (e.g. a6c together with
        d_delta_t_nqc).

        Parameters/Returns: same (opts, mm_opt) contract as the other
        backends; the full posterior lives in self.sampler_extras (see
        _summarize_posterior) after this returns.
        """
        if len(kys) < 2:
            raise ValueError(
                f"The 'nessai' backend requires at least 2 optimization "
                f"variables (got {len(kys)}: {kys}) -- nessai's normalising-"
                f"flow proposals are not designed for one-dimensional models. "
                f"Use 'dynesty' for single-parameter Bayesian fits."
            )

        from nessai.model import Model as NessaiModel
        from nessai.flowsampler import FlowSampler
        import nessai.posterior as nessai_posterior

        optimizer_self = self
        bounds_dict = {
            ky: [float(bounds_array[i][0]), float(bounds_array[i][1])]
            for i, ky in enumerate(kys)
        }

        class _MismatchModel(NessaiModel):
            def __init__(self):
                self.names = list(kys)
                self.bounds = bounds_dict
                # No parallelism / vectorisation: f (and hence the EOB model)
                # is not guaranteed picklable, and TEOB is a C extension
                # underneath -- see calibration upgrade plan, Stage 3.D.
                self.allow_vectorised = False
                self.n_pool = None
                super().__init__()

            def log_prior(self, x):
                return np.log(self.in_bounds(x), dtype=float)

            def log_likelihood(self, x):
                vs = [x[ky] for ky in self.names]
                return optimizer_self._sampler_log_likelihood(vs, self.names)

        model = _MismatchModel()

        output_dir = self.minimizer.get("output_dir")
        if not output_dir:
            raise ValueError(
                "The 'nessai' backend requires minimizer['output_dir'] -- "
                "nessai writes checkpoints/results to disk and must never "
                "default to CWD (shared under Condor initialdir)."
            )
        nlive = self.minimizer.get("nlive", 500)
        seed = self.minimizer.get("opt_seed", 190521)
        resume = self.minimizer.get("resume", True)

        sampler = FlowSampler(
            model,
            output=output_dir,
            nlive=nlive,
            seed=seed,
            resume=resume,
            resume_file="nessai_resume.pkl",
        )
        sampler.run()

        ns = sampler.nested_samples
        maxL_idx = ns["logL"].argmax()
        opts = {ky: float(ns[ky][maxL_idx]) for ky in kys}
        # mm_opt must be a real mismatch (not a log-likelihood) for
        # comparability with the other backends -- recompute it directly at
        # the maxL point rather than trying to invert the likelihood.
        eob_opt = self.generate_EOB(model=self.model, ICs=opts)
        mm_opt = (
            self.objective_mismatch(eob_opt, verbose=False, iter_loop=False)
            if eob_opt is not None
            else 1.0
        )

        posterior_samples = nessai_posterior.draw_posterior_samples(ns, nlive=nlive)
        posterior_array = np.column_stack([posterior_samples[ky] for ky in kys])

        self.sampler_extras = self._summarize_posterior(
            kys,
            posterior_array,
            "nessai",
            output_dir=output_dir,
            log_evidence=sampler.log_evidence,
            log_evidence_err=sampler.log_evidence_error,
            n_likelihood_evaluations=model.likelihood_evaluations,
        )
        self.sampler_extras["likelihood_settings"] = dict(self.likelihood_settings)
        self.sampler_extras["n_failed_likelihood_evals"] = (
            self._n_failed_likelihood_evals
        )

        return opts, mm_opt

    def __minimize_zeus__(self, f, x0, bounds_array, kys):
        """Minimize with zeus (ensemble slice sampling MCMC).

        Unlike the nested samplers (dynesty/nessai), which must explore the
        full prior volume from scratch, zeus is warm-started directly:
        walkers are initialized in a small ball around the box center
        (self._bounds_reference, which fit_parameter.py's warm-start pass
        already centers on a cheap point estimate) rather than drawn
        uniformly across the whole prior box. See calibration upgrade plan,
        Stage 3 prototype notes -- this is the intended fix for nested
        sampling paying to re-discover a mode we already located cheaply.
        No evidence estimate (ensemble MCMC doesn't produce one); use
        dynesty/nessai/pocomc if evidence is needed.
        """
        import zeus

        ndim = len(kys)
        nwalkers = self.minimizer.get("nwalkers", max(4 * ndim, 8))
        nsteps = self.minimizer.get("nsteps", 3000)
        nburn = self.minimizer.get("nburn", nsteps // 2)
        init_frac = self.minimizer.get("init_frac", 0.1)
        seed = self.minimizer.get("opt_seed", 190521)
        progress = self.minimizer.get("print_progress", True)
        rng = np.random.default_rng(seed)

        lo = bounds_array[:, 0]
        hi = bounds_array[:, 1]
        center = np.array(
            [self._bounds_reference.get(ky, x0[i]) for i, ky in enumerate(kys)]
        )
        spread = init_frac * (hi - lo)

        def logpost(x):
            if np.any(x < lo) or np.any(x > hi):
                return -np.inf
            logl = self._sampler_log_likelihood(x, kys)
            return logl if np.isfinite(logl) else -np.inf

        p0 = center + spread * rng.standard_normal((nwalkers, ndim))
        p0 = np.clip(p0, lo, hi)

        sampler = zeus.EnsembleSampler(nwalkers, ndim, logpost, verbose=progress)
        sampler.run_mcmc(p0, nsteps)

        chain = sampler.get_chain(discard=nburn, flat=True)
        logprob = sampler.get_log_prob(discard=nburn, flat=True)

        maxL_idx = int(np.argmax(logprob))
        opts = {kys[i]: float(chain[maxL_idx][i]) for i in range(ndim)}
        eob_opt = self.generate_EOB(model=self.model, ICs=opts)
        mm_opt = (
            self.objective_mismatch(eob_opt, verbose=False, iter_loop=False)
            if eob_opt is not None
            else 1.0
        )

        output_dir = self.minimizer.get("output_dir")
        self.sampler_extras = self._summarize_posterior(
            kys,
            chain,
            "zeus",
            output_dir=output_dir,
            log_evidence=None,
            log_evidence_err=None,
            n_likelihood_evaluations=int(getattr(sampler, "ncall", nwalkers * nsteps)),
        )
        self.sampler_extras["likelihood_settings"] = dict(self.likelihood_settings)
        self.sampler_extras["n_failed_likelihood_evals"] = (
            self._n_failed_likelihood_evals
        )
        self.sampler_extras["nwalkers"] = nwalkers
        self.sampler_extras["nburn"] = nburn

        return opts, mm_opt

    def __minimize_pocomc__(self, f, x0, bounds_array, kys):
        """Minimize with pocoMC (preconditioned Monte Carlo / sequential
        Monte Carlo with normalising-flow preconditioning).

        Purpose-built for expensive likelihoods, and unlike zeus still
        produces an evidence estimate. Initial particles are drawn from the
        uniform prior over bounds_array (already the warm-started box from
        fit_parameter.py) -- pocoMC's SMC annealing schedule adapts from
        there rather than needing an explicit x0 seed the way zeus does.
        """
        import pocomc as pc
        from scipy.stats import uniform as scipy_uniform

        ndim = len(kys)
        lo = bounds_array[:, 0]
        hi = bounds_array[:, 1]

        prior = pc.Prior(
            [scipy_uniform(loc=lo[i], scale=hi[i] - lo[i]) for i in range(ndim)]
        )

        def loglike(x):
            logl = self._sampler_log_likelihood(x, kys)
            return logl if np.isfinite(logl) else -np.inf

        n_effective = self.minimizer.get("n_effective", 64)
        n_active = self.minimizer.get("n_active", 32)
        n_total = self.minimizer.get("n_total", 1024)
        # n_evidence=0 skips pocoMC's dedicated importance-sampling evidence
        # refinement step -- measured at exactly half the total likelihood
        # calls in a real run (1024 of 2176). Calibration wants posterior
        # width, not model comparison, so this is off by default; evidence()
        # still returns a (less precise, no error bar) SMC-based logz even
        # with n_evidence=0 -- see calibration upgrade plan, Stage 3.E.
        n_evidence = self.minimizer.get("n_evidence", 0)
        seed = self.minimizer.get("opt_seed", 190521)
        progress = self.minimizer.get("print_progress", True)
        output_dir = self.minimizer.get("output_dir")
        # Normalising-flow preconditioning exists to remove parameter
        # correlations; with a single parameter (or few, uncorrelated ones)
        # there is nothing for it to remove, so it plausibly only costs CPU
        # -- exposed for testing, library default (True) unchanged for now.
        precondition = self.minimizer.get("precondition", True)

        sampler = pc.Sampler(
            prior=prior,
            likelihood=loglike,
            n_dim=ndim,
            n_effective=n_effective,
            n_active=n_active,
            vectorize=False,
            precondition=precondition,
            random_state=seed,
            output_dir=output_dir,
        )
        sampler.run(n_total=n_total, n_evidence=n_evidence, progress=progress)

        # resample=True already returns equal-weight samples (no weights
        # array) -- resample=False would instead give (samples, weights,
        # logl, logp) with importance weights still attached.
        samples, logl, logp = sampler.posterior(resample=True)
        samples = np.asarray(samples)

        maxL_idx = int(np.argmax(logl))
        opts = {kys[i]: float(samples[maxL_idx][i]) for i in range(ndim)}
        eob_opt = self.generate_EOB(model=self.model, ICs=opts)
        mm_opt = (
            self.objective_mismatch(eob_opt, verbose=False, iter_loop=False)
            if eob_opt is not None
            else 1.0
        )

        log_evidence, log_evidence_err = sampler.evidence()

        self.sampler_extras = self._summarize_posterior(
            kys,
            samples,
            "pocomc",
            output_dir=output_dir,
            log_evidence=log_evidence,
            log_evidence_err=log_evidence_err,
            n_likelihood_evaluations=int(getattr(sampler, "calls", n_total)),
        )
        self.sampler_extras["likelihood_settings"] = dict(self.likelihood_settings)
        self.sampler_extras["n_failed_likelihood_evals"] = (
            self._n_failed_likelihood_evals
        )

        return opts, mm_opt

    def __minimize_grid__(self, f, x0, bounds_array, kys):
        """
        Minimization via Grid/quadrature backend.

        This backend is designed for low-dimensional problems where a full grid evaluation is computationally feasible.


        Only `ndim==1` is handled by the quadrature below; a joint (2-D)
        posterior is left to the `gp_surrogate` backend -- interpolating a
        2-D scattered node set correctly needs a triangulation or a GP, not
        the 1-D trapezoid rule used here.
        """
        ndim = len(kys)
        lo = np.asarray(bounds_array[:, 0], dtype=float)
        hi = np.asarray(bounds_array[:, 1], dtype=float)

        # we first run a scalar minimization to find the best internal estimate of the optimum.
        if ndim == 1:
            opts_scalar, mm_opt = self.__minimize_scalar_(f, x0, bounds_array, kys)
            p_best = np.array([opts_scalar[kys[0]]])
            mm_opt = float(mm_opt)
        else:
            raise NotImplementedError("For the moment, only ndim==1 is implemented.")

        opts = {kys[i]: float(p_best[i]) for i in range(ndim)}
        _, logl_best = self._eval_mm_and_logl(p_best, kys)

        # Set up the coarse and fine grid parameters.
        n_coarse = int(self.minimizer.get("n_coarse", 64))
        n_fine = int(self.minimizer.get("n_fine", 32))
        adapt = self.minimizer.get("adapt", True)
        expand_below = float(self.minimizer.get("delta_logl_expand", 10.0))
        contract_above = float(self.minimizer.get("delta_logl_contract", 50.0))
        max_adapt_iter = int(self.minimizer.get("max_adapt_iter", 8))

        # Determine the initial core half-width for the fine grid around the optimum.
        half_box = 0.5 * (hi - lo)
        init_frac = float(self.minimizer.get("core_half_width_frac", 0.1))
        core_init = self.minimizer.get("core_half_width_init")
        if core_init is None:
            core_half = init_frac * half_box
        else:
            core_half = np.full(ndim, float(core_init))
        core_half = np.minimum(core_half, half_box)

        # Adapt the core half-width based on the log-likelihood drop at the edges.
        if adapt:
            for _ in range(max_adapt_iter):
                changed = False
                for d in range(ndim):
                    probe_lo = p_best.copy()
                    probe_lo[d] = np.clip(p_best[d] - core_half[d], lo[d], hi[d])
                    probe_hi = p_best.copy()
                    probe_hi[d] = np.clip(p_best[d] + core_half[d], lo[d], hi[d])
                    _, logl_lo = self._eval_mm_and_logl(probe_lo, kys)
                    _, logl_hi = self._eval_mm_and_logl(probe_hi, kys)
                    d_logl = logl_best - min(logl_lo, logl_hi)
                    if not np.isfinite(d_logl):
                        d_logl = np.inf
                    if d_logl < expand_below and core_half[d] < half_box[d]:
                        core_half[d] = min(core_half[d] * 1.5, half_box[d])
                        changed = True
                    elif d_logl > contract_above and core_half[d] > 0:
                        core_half[d] = core_half[d] / 1.5
                        changed = True
                if not changed:
                    break

        # Warn if the core half-width has been capped at the level-0 box.
        capped = core_half >= half_box - 1e-12 * np.maximum(half_box, 1.0)
        if np.any(capped):
            logging.warning(
                "grid backend: core half-width capped at the level-0 box on "
                f"axis(es) {[kys[d] for d in range(ndim) if capped[d]]} -- the "
                "posterior is prior-box-dominated, not resolved by this box."
            )

        coarse_axes = [np.linspace(lo[d], hi[d], n_coarse) for d in range(ndim)]
        fine_axes = [
            np.linspace(p_best[d] - core_half[d], p_best[d] + core_half[d], n_fine)
            for d in range(ndim)
        ]

        # Construct the full tensor grid for both coarse and fine levels.
        def _tensor_grid(axes):
            mesh = np.meshgrid(*axes, indexing="ij")
            return np.stack([m.ravel() for m in mesh], axis=-1)

        level0_points = _tensor_grid(coarse_axes)
        level1_points = _tensor_grid(fine_axes)
        all_points_raw = np.concatenate([level0_points, level1_points], axis=0)
        grid_level_raw = np.concatenate(
            [np.zeros(len(level0_points)), np.ones(len(level1_points))]
        )

        all_points, unique_idx = np.unique(all_points_raw, axis=0, return_index=True)
        grid_level = grid_level_raw[unique_idx]

        n_points = all_points.shape[0]
        grid_mm = np.full(n_points, np.nan)
        grid_logl = np.full(n_points, -np.inf)
        for i in range(n_points):
            grid_mm[i], grid_logl[i] = self._eval_mm_and_logl(all_points[i], kys)

        argmax_idx = int(np.nanargmax(grid_logl))
        grid_argmax = all_points[argmax_idx]
        fine_spacing = np.array(
            [
                (fine_axes[d][-1] - fine_axes[d][0]) / max(n_fine - 1, 1)
                for d in range(ndim)
            ]
        )
        if np.any(np.abs(grid_argmax - p_best) > fine_spacing):
            logging.warning(
                "grid backend: grid argmax %s disagrees with the internal "
                "point estimate %s by more than one fine-grid spacing %s -- "
                "check point-estimate convergence and grid coverage.",
                grid_argmax.tolist(),
                p_best.tolist(),
                fine_spacing.tolist(),
            )

        # Sort the grid points by the first dimension to prepare for trapezoidal integration.
        order = np.argsort(all_points[:, 0])
        x_sorted = all_points[order, 0]
        logl_sorted = grid_logl[order]
        finite = np.isfinite(logl_sorted)
        if not np.any(finite):
            raise RuntimeError(
                "grid backend: every node had -inf log-likelihood (EOB failed "
                "everywhere in the box) -- cannot build a posterior."
            )
        logl_max = np.max(logl_sorted[finite])
        density = np.where(finite, np.exp(logl_sorted - logl_max), 0.0)

        box_evidence = np.trapz(density, x_sorted)
        log_evidence = float(logl_max + np.log(box_evidence))

        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(x_sorted))]
        )
        cdf = cdf / cdf[-1]
        # Draw posterior samples by inverting the CDF.
        n_samples = int(self.minimizer.get("n_posterior_samples", 5000))
        u = np.random.default_rng(self.minimizer.get("opt_seed", 190521)).uniform(
            0.0, 1.0, size=n_samples
        )
        posterior_samples = np.interp(u, cdf, x_sorted).reshape(-1, 1)

        output_dir = self.minimizer.get("output_dir")
        self.sampler_extras = self._summarize_posterior(
            kys,
            posterior_samples,
            "grid",
            output_dir=output_dir,
            log_evidence=log_evidence,
            log_evidence_err=None,
            n_likelihood_evaluations=n_points,
        )
        self.sampler_extras["likelihood_settings"] = dict(self.likelihood_settings)
        self.sampler_extras["n_failed_likelihood_evals"] = (
            self._n_failed_likelihood_evals
        )
        self.sampler_extras["grid_points"] = all_points[:, 0].tolist()
        self.sampler_extras["grid_mm"] = grid_mm.tolist()
        self.sampler_extras["grid_logl"] = grid_logl.tolist()
        self.sampler_extras["grid_level"] = grid_level.tolist()
        self.sampler_extras["grid_argmax"] = grid_argmax.tolist()
        self.sampler_extras["grid_core_half_width"] = core_half.tolist()
        self.sampler_extras["log_evidence_note"] = (
            "integral over the grid box, not the prior -- not comparable to "
            "dynesty's or pocomc's log_evidence"
        )

        # Generate & store the posterior density curve for visualization and further analysis.
        n_curve_points = int(self.minimizer.get("n_curve_points", 400))
        x_out = np.linspace(x_sorted[0], x_sorted[-1], n_curve_points)
        pdf_exact_out = np.interp(x_out, x_sorted, density)
        pdf_exact_out = pdf_exact_out / np.trapz(pdf_exact_out, x_out)

        pdf_kde_out = None
        kde = None
        finite_weight = density > 0
        if np.count_nonzero(finite_weight) >= 2 and np.ptp(x_sorted[finite_weight]) > 0:
            try:
                from scipy.stats import gaussian_kde

                kde = gaussian_kde(
                    x_sorted[finite_weight],
                    weights=density[finite_weight],
                    bw_method=self.minimizer.get("kde_bw_method"),
                )
            except Exception as exc:
                logging.warning(
                    f"grid backend: weighted-KDE construction failed ({exc}) -- "
                    "posterior_density['pdf_kde'] will be null."
                )
                self.sampler_extras["kde_error"] = str(exc)
        else:
            logging.warning(
                "grid backend: fewer than 2 finite-density nodes -- cannot build "
                "a KDE; posterior_density['pdf_kde'] will be null."
            )

        if kde is not None:
            pdf_kde_out = kde.evaluate(x_out)

            n_kde_samples = int(self.minimizer.get("n_posterior_samples", 5000))
            kde_samples = np.asarray(
                kde.resample(n_kde_samples, seed=self.minimizer.get("opt_seed", 190521))
            ).T

            # Compare the standard deviation of the KDE samples to the exact quadrature samples to check for consistency.
            exact_sigma_check = float(np.std(posterior_samples[:, 0]))
            kde_sigma_check = float(np.std(kde_samples[:, 0]))
            if exact_sigma_check > 0:
                rel_diff = abs(kde_sigma_check - exact_sigma_check) / exact_sigma_check
                if rel_diff > 0.2:
                    logging.warning(
                        f"grid backend: weighted-KDE sigma ({kde_sigma_check:.4g}) "
                        f"disagrees with the exact quadrature sigma "
                        f"({exact_sigma_check:.4g}) by {rel_diff:.0%} -- check "
                        "minimizer['kde_bw_method'] before trusting kde_* outputs."
                    )

            kde_output_dir = os.path.join(output_dir, "kde") if output_dir else None
            kde_extras = self._summarize_posterior(
                kys, kde_samples, "grid_kde", output_dir=kde_output_dir
            )
            for ky, val in kde_extras.items():
                if ky == "sampler":
                    continue
                self.sampler_extras[f"kde_{ky}"] = val
            self.sampler_extras["kde_bandwidth_factor"] = float(kde.factor)
            self.sampler_extras["kde_neff"] = float(kde.neff)

        self.sampler_extras["posterior_density"] = {
            "x": x_out.tolist(),
            "pdf_exact": pdf_exact_out.tolist(),
            "pdf_kde": pdf_kde_out.tolist() if pdf_kde_out is not None else None,
        }

        return opts, mm_opt

    def __minimize_gp_surrogate__(self, f, x0, bounds_array, kys):
        """
        Minimize with a Gaussian Process surrogate using the GPry package.
        This is an active-learning approach where a Gaussian Process surrogate is used to approximate the objective function, reducing the number of expensive true evaluations needed.
        For more info see: https://gpry.readthedocs.io/en/latest/
        """
        try:
            from gpry.run import Runner
        except ImportError as exc:
            raise ImportError(
                "minimizer kind='gp_surrogate' requires the 'gpry' package "
                "(pip install gpry) -- not installed in this environment."
            ) from exc
        from dynesty.utils import resample_equal

        ndim = len(kys)
        bounds = [
            [float(bounds_array[i][0]), float(bounds_array[i][1])] for i in range(ndim)
        ]

        n_calls = 0
        best = {"logl": -np.inf, "mm": None, "x": None}

        def loglike(x):
            nonlocal n_calls
            n_calls += 1
            # Ensure x is always at least 1-dimensional, even if GPry passes a scalar for ndim==1.
            x_arr = np.atleast_1d(x)
            mm, logl = self._eval_mm_and_logl(x_arr, kys)
            if np.isfinite(logl) and logl > best["logl"]:
                best["logl"], best["mm"], best["x"] = logl, mm, x_arr.copy()
            return logl if np.isfinite(logl) else -np.inf

        output_dir = self.minimizer.get("output_dir")
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            mc_output = os.path.join(output_dir, "gpry_mc")
        else:
            # If no output_dir is specified, create a temporary directory for GPry's MC output.
            mc_output = tempfile.mkdtemp(prefix="gpry_mc_")

        seed = self.minimizer.get("opt_seed", 190521)
        verbose = self.minimizer.get("gp_verbose", 1)
        gp_options = self.minimizer.get("gp_options")

        runner = Runner(
            loglike=loglike,
            bounds=bounds,
            params=list(kys),
            options=gp_options,
            checkpoint=None,  # no checkpointing to avoid pickling issues with self
            seed=seed,
            plots=False,
            verbose=verbose,
        )
        runner.run()

        # The opts/mm_opt returned here correspond to the best true evaluation seen during the surrogate training, not the surrogate's own approximate optimum.
        opts = {kys[i]: float(best["x"][i]) for i in range(ndim)}
        mm_opt = float(best["mm"])
        training_df = runner.surrogate.training_set_as_df()

        # Generate Monte Carlo samples from the surrogate posterior using nested sampling.
        # no further true likelihood evaluations are performed; all samples come from the surrogate.
        runner.generate_mc_sample(sampler="nested", output=mc_output)
        mc = runner.last_mc_samples()
        mc_X = np.atleast_2d(mc["X"])
        mc_w = mc["w"]
        if mc_w is None:
            posterior_samples = mc_X
        else:
            posterior_samples = resample_equal(mc_X, np.asarray(mc_w, dtype=float))

        log_evidence, log_evidence_err = runner.last_mc_logZ()

        self.sampler_extras = self._summarize_posterior(
            kys,
            posterior_samples,
            "gp_surrogate",
            output_dir=output_dir,
            log_evidence=log_evidence,
            log_evidence_err=log_evidence_err,
            n_likelihood_evaluations=n_calls,
        )
        self.sampler_extras["likelihood_settings"] = dict(self.likelihood_settings)
        self.sampler_extras["n_failed_likelihood_evals"] = (
            self._n_failed_likelihood_evals
        )
        self.sampler_extras["n_gp_training_points"] = int(len(training_df))

        return opts, mm_opt

    def __minimize_nautilus__(self, f, x0, bounds_array, kys):
        """
        Minimise/characterise via nautilus (neural-network-boosted
        importance nested sampling).
        For more info: https://nautilus-sampler.readthedocs.io/en/latest/

        NOTE: this doesn't work if the number of optimization variables is less than 2.
        """
        if len(kys) < 2:
            raise ValueError(
                "minimizer kind='nautilus' requires at least 2 optimization variables."
            )
        try:
            from nautilus import Sampler, Prior
        except ImportError as exc:
            raise ImportError(
                "Install the 'nautilus-sampler' package (pip install nautilus-sampler) -- not installed in this environment."
            ) from exc
        from dynesty.utils import resample_equal

        ndim = len(kys)
        prior = Prior()
        for i, ky in enumerate(kys):
            prior.add_parameter(
                ky, dist=(float(bounds_array[i][0]), float(bounds_array[i][1]))
            )

        n_calls = 0

        def loglike(d):
            # pass_dict=True (forced below, since prior is a nautilus.Prior
            # instance) -- d is {param_name: value}, not a positional array.
            nonlocal n_calls
            n_calls += 1
            x = np.array([d[ky] for ky in kys])
            logl = self._sampler_log_likelihood(x, kys)
            return logl if np.isfinite(logl) else -np.inf

        n_live = int(self.minimizer.get("n_live", 500))
        n_eff = float(self.minimizer.get("n_eff", 1000.0))
        seed = self.minimizer.get("opt_seed", 190521)
        verbose = self.minimizer.get("print_progress", True)

        sampler = Sampler(
            prior,
            loglike,
            n_live=n_live,
            vectorized=False,
            pass_dict=True,
            seed=seed,
        )
        sampler.run(n_eff=n_eff, verbose=verbose)

        points, log_w, log_l = sampler.posterior()
        maxL_idx = int(np.argmax(log_l))
        opts = {kys[i]: float(points[maxL_idx, i]) for i in range(ndim)}
        eob_opt = self.generate_EOB(model=self.model, ICs=opts)
        mm_opt = (
            self.objective_mismatch(eob_opt, verbose=False, iter_loop=False)
            if eob_opt is not None
            else 1.0
        )

        # posterior() returns importance-weighted samples (log_w), the same
        # unequal-weight situation as raw dynesty output -- reuse the same
        # resample_equal utility already used there rather than duplicating
        # the equal-weighting logic.
        weights = np.exp(log_w - np.max(log_w))
        posterior_samples = resample_equal(points, weights)

        self.sampler_extras = self._summarize_posterior(
            kys,
            posterior_samples,
            "nautilus",
            output_dir=self.minimizer.get("output_dir"),
            log_evidence=float(sampler.log_z),
            log_evidence_err=None,  # nautilus exposes no evidence error bar
            n_likelihood_evaluations=n_calls,
        )
        self.sampler_extras["likelihood_settings"] = dict(self.likelihood_settings)
        self.sampler_extras["n_failed_likelihood_evals"] = (
            self._n_failed_likelihood_evals
        )

        return opts, mm_opt
