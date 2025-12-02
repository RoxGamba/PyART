import sys, os, argparse, subprocess, matplotlib, time
import numpy as np
import matplotlib.pyplot as plt
from PyART.catalogs import sxs
from PyART.models import teob
from PyART.analysis.match import Matcher
from PyART.utils import utils as ut

matplotlib.rc("text", usetex=True)

parser = argparse.ArgumentParser()

parser.add_argument("-id", "--sxs_id", default=180, help="SXS ID")
parser.add_argument("--mass_min", type=float, default=10, help="Minimum mass (Msun)")
parser.add_argument("--mass_max", type=float, default=200, help="Minimum mass (Msun)")
parser.add_argument("-n", "--mass_num", type=int, default=20, help="Number of masses")
parser.add_argument("--f1", type=float, default=None, help="Initial freq for mm")
parser.add_argument("--f2", type=float, default=2048, help="Final freq for mm")
parser.add_argument(
    "--f0_from_merger", action="store_true", help="Get initial freq for mm from merger"
)
parser.add_argument("--taper", type=str, default="sigmoid", help="Kind of taper")
parser.add_argument("--taper_alpha", type=float, default=0.20, help="Taper alpha")
parser.add_argument("--taper_start", type=float, default=0.05, help="Taper start")
parser.add_argument("--taper_end", type=float, default=None, help="Taper end")
parser.add_argument("-d", "--debug", action="store_true", help="Show debug plots")
parser.add_argument("--cut", action="store_true", help="Cut waves before mm")
parser.add_argument("--no_plot", action="store_true", help="Avoid mm-plot")
args = parser.parse_args()

repo_path = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sxs_path = os.path.join(repo_path, "examples/local_data/sxs/")

# load (or download) SXS data
sxs_id = f"{args.sxs_id:04}"  # e.g.0180
nr = sxs.Waveform_SXS(
    path=sxs_path, download=True, ID=sxs_id, order="Extrapolated_N3.dir", ellmax=7
)
nr.cut(300)
# nr.compute_hphc()

# read SXS-meta
q = nr.metadata["q"]
M = nr.metadata["M"]
chi1z = nr.metadata["chi1z"]
chi2z = nr.metadata["chi2z"]
f0 = nr.metadata["f0"]

print("-----------------------------")
print(f"Mismatch for SXS:{sxs_id}")
print("-----------------------------")
print(f"q     : {q:.5f}")
print(f"M     : {M:.5f}")
print(f"chi1z : {chi1z:.5f}")
print(f"chi2z : {chi2z:.5f}")
print(f"f0    : {f0:.5f}")
print("-----------------------------")

# compute corresponding EOB waveform
Tmax = 1e4
srate = 8192
eobpars = {
    "M": M,
    "q": q,
    "chi1": chi1z,
    "chi2": chi2z,
    "LambdaAl2": 0.0,
    "LambdaBl2": 0.0,
    "distance": 1.0,
    "initial_frequency": 0.9 * f0,  # magic number for 0180: 0.988*f0,
    "use_geometric_units": "yes",
    "interp_uniform_grid": "yes",
    "domain": 0,
    "srate_interp": srate,
    "inclination": 0.0,
    "output_hpc": "no",
    "use_mode_lm": [1],  # List of modes to use
    "arg_out": "yes",  # output dynamics and hlm in addition to h+, hx
    "ecc": 1e-8,
    "r_hyp": 0.0,
    "H_hyp": 0.0,
    "j_hyp": 0.0,
    "coalescence_angle": 0.0,
    "df": 1.0 / 128.0,
    "anomaly": np.pi,
    "spin_flx": "EOB",
    "spin_interp_domain": 0,
}
eob = teob.Waveform_EOB(pars=eobpars)
eob.multiply_by(var=["hlm"], factor=q / (1 + q) ** 2)

# plt.figure
# plt.plot(eob.u, eob.hlm[(2,2)]['A'])
# plt.show()

# nr_mrg,_,_,_  = nr.find_max()
# eob_mrg,_,_,_ = eob.find_max()
# plt.figure
# plt.plot(nr.u-nr_mrg, nr.hlm[(2,2)]['real'])
# plt.plot(eob.u-eob_mrg, eob.hlm[(2,2)]['real'])
# plt.show()

# compute (2,2) mismatches for different masses
masses = np.linspace(args.mass_min, args.mass_max, num=args.mass_num)
mm = masses * 0.0
t0 = time.perf_counter()
for i, M in enumerate(masses):
    if args.f1 is None:
        f0_mm = 1.25 * f0 / (M * ut.Msun)
    else:
        f0_mm = args.f1
    matcher = Matcher(
        nr,
        eob,
        settings={
            #'kind':'single-mode',
            "pre_align": True,
            "initial_frequency_mm": f0_mm,
            "final_frequency_mm": args.f2,
            "tlen": len(nr.u),
            "dt": 1 / srate,
            "M": M,
            "resize_factor": 4,
            "modes-or-pol": "modes",
            "modes": [(2, 2)],
            "pad_end_frac": 0.5,
            "taper_alpha": args.taper_alpha,
            "taper_start": args.taper_start,
            "taper_end": args.taper_end,
            "taper": args.taper,
            "cut_longer": args.cut,
            "debug": args.debug,
            "f0_from_merger": args.f0_from_merger,
        },
    )
    mm[i] = matcher.mismatch

    print(f"mass:{M:8.2f}, f0_mm:{f0_mm:8.4f} Hz, mm: {mm[i]:.3e}")

print("Elapsed time [s]: ", time.perf_counter() - t0)

if not args.no_plot and args.mass_num > 1:
    plt.figure(figsize=(9, 6))
    plt.plot(masses, mm)
    plt.yscale("log")
    plt.xlabel(r"$M_\odot$", fontsize=25)
    plt.ylabel(r"$\bar{\cal{F}}$", fontsize=25)
    plt.ylim(1e-4, 1e-1)
    plt.grid()
    plt.show()
