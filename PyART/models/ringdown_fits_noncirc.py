"""
Functions to read and use the Noncircular fits
from Carullo+23
https://arxiv.org/abs/2309.07228

Mostly taken from Greg's repo, without all the frills
https://github.com/GCArullo/noncircular_BBH_fits
"""

import pandas as pd
import logging
import numpy as np
import os

fit_dim = 4  # Number of free coefficient for each fitting variable


def select_fitting_quantities(dataset_type, quantity_to_fit):

    if quantity_to_fit == "A_peak22":
        fitting_quantities_base_list = ["b_massless_EOB", "Heff_til-b_massless_EOB"]
    elif quantity_to_fit == "omega_peak22":
        fitting_quantities_base_list = ["b_massless_EOB", "Heff_til-b_massless_EOB"]
    elif quantity_to_fit == "Mf":
        fitting_quantities_base_list = ["Heff_til", "Heff_til-Jmrg_til"]
    elif quantity_to_fit == "af":
        fitting_quantities_base_list = ["Jmrg_til", "Heff_til-Jmrg_til"]

    # In case we are fitting non-equal mass data, add nu-dependence to each fitting quantity.
    if dataset_type == "non-spinning":
        fitting_quantities_strings_list = []
        for fx in fitting_quantities_base_list:
            fitting_quantities_strings_list.append("nu-" + fx)
    elif dataset_type == "aligned-spins-equal-mass":
        fitting_quantities_strings_list = []
        for fx in fitting_quantities_base_list:
            fitting_quantities_strings_list.append("chieff-" + fx)
    else:
        fitting_quantities_strings_list = fitting_quantities_base_list

    return fitting_quantities_strings_list


def select_template_model(dataset_type):

    # The factorised model applies only to the unequal mass case
    if dataset_type == "non-spinning-equal-mass":
        template_model = "rational"
    elif dataset_type == "non-spinning":
        template_model = "factorised-nu"
    elif dataset_type == "aligned-spins-equal-mass":
        template_model = "rational"

    return template_model


def read_fit_coefficients(
    quantity_to_fit,
    fitting_quantities_string,
    template_model,
    fit_dim,
    catalogs=["RIT", "SXS", "ET"],
    dataset_type=["non-spinning-equal-mass"],
    coeffs_dir="./noncircular_BBH_fits/Fitting_coefficients",
):

    catalogs_string = ""
    for catalog_y in catalogs:
        catalogs_string += catalog_y + "-"
    catalogs_string = catalogs_string[:-1]
    path = os.path.join(
        coeffs_dir,
        f"Fitting_coefficients_{dataset_type}_{catalogs_string}_{template_model}_{fit_dim}_{fitting_quantities_string}_{quantity_to_fit}.csv",
    )
    coeffs = pd.read_csv(path)["coeffs"]
    coeffs = np.array(coeffs)

    return coeffs


def template(coeffs, fitting_quantities_dict, template_model="rational"):

    any_quantity_name = list(fitting_quantities_dict.keys())[0]
    len_data = len(fitting_quantities_dict[any_quantity_name])
    ones_len_data = [1.0] * len_data
    result = [coeffs[0]] * len_data
    len_coeffs = len(coeffs) - 1
    fitting_quantities_dict_loop = list(fitting_quantities_dict.keys())
    if "factorised-nu" in template_model:
        fitting_quantities_dict_loop.remove("nu")
        single_var_coeffs_len = int(
            int(len_coeffs / len(fitting_quantities_dict_loop)) / 2
        )
    else:
        single_var_coeffs_len = int(len_coeffs / len(fitting_quantities_dict_loop))
    first_half_vec_len = int((single_var_coeffs_len) / 2)

    # Loop on the fitting quantities
    for i, key_x in enumerate(fitting_quantities_dict_loop):

        # Construct a rational function per fitting quantity
        if template_model == "rational":

            # With this structure, the first block of length single_var_coeffs_len are the coefficients of the first variable, the second block are the ones of the second, and so on.

            result_i_num = ones_len_data + np.sum(
                [
                    coeffs[j]
                    * fitting_quantities_dict[key_x]
                    ** (j - (i * single_var_coeffs_len))
                    for j in range(
                        1 + i * single_var_coeffs_len,
                        (first_half_vec_len + 1) + (i * single_var_coeffs_len),
                    )
                ],
                axis=0,
            )
            result_i_den = ones_len_data + np.sum(
                [
                    coeffs[j]
                    * fitting_quantities_dict[key_x]
                    ** (j - (i * single_var_coeffs_len) - first_half_vec_len)
                    for j in range(
                        (first_half_vec_len + 1) + i * single_var_coeffs_len,
                        ((i + 1) * single_var_coeffs_len) + 1,
                    )
                ],
                axis=0,
            )

        # Construct a rational function per fitting quantity, except for the mass ratio, which is folded-in through X
        # With this structure, the first half of the coefficients are the X=0, while the second half are the X=1 in reverse order

        elif "factorised-nu" in template_model:

            if key_x == "nu":
                continue

            X = 1 - 4.0 * fitting_quantities_dict["nu"]

            if template_model == "factorised-nu":

                result_i_num = ones_len_data + np.sum(
                    [
                        coeffs[j]
                        * fitting_quantities_dict[key_x]
                        ** (j - (i * single_var_coeffs_len))
                        * (1.0 + coeffs[-j] * X)
                        for j in range(
                            1 + i * single_var_coeffs_len,
                            (first_half_vec_len + 1) + (i * single_var_coeffs_len),
                        )
                    ],
                    axis=0,
                )
                result_i_den = ones_len_data + np.sum(
                    [
                        coeffs[j]
                        * fitting_quantities_dict[key_x]
                        ** (j - (i * single_var_coeffs_len) - first_half_vec_len)
                        * (1.0 + coeffs[-j] * X)
                        for j in range(
                            (first_half_vec_len + 1) + i * single_var_coeffs_len,
                            ((i + 1) * single_var_coeffs_len) + 1,
                        )
                    ],
                    axis=0,
                )

        result *= result_i_num / result_i_den

    return result


def eval_fit(
    quantity_to_fit,
    fitting_quantities_dict,
    fitting_quantities_string,
    dataset="non-spinning",
    databases=["RIT", "SXS", "ET"],
    verbose=False,
):
    """
    Perform a fit of a given quantity based on a chosen dataset.
    Choose between:
        - quantities_to_fit    = ['A_peak22', 'omega_peak22', 'Mf', 'af']
        - dataset_types        = ['aligned-spins-equal-mass','non-spinning-equal-mass', 'non-spinning']
    """

    fitting_qs_d = fitting_quantities_dict.copy()
    for key in list(fitting_qs_d.keys()):
        if key not in fitting_quantities_string:
            fitting_qs_d.pop(key, None)

    # Find the independent vars allowed
    if verbose:
        logging.info(f"Fitting quantity:\t\t {quantity_to_fit}")
        fitting_qs = select_fitting_quantities(dataset, quantity_to_fit)
        logging.info(f"Possible independent vars:\t {fitting_qs}")
        logging.info(f"Chosen independent vars:\t {list(fitting_qs_d.keys())}")

    # select template
    template_model = select_template_model(dataset)
    if verbose:
        logging.info(f"Template model:\t\t\t {template_model}")

    try:
        coeffs = read_fit_coefficients(
            quantity_to_fit,
            fitting_quantities_string,
            template_model,
            fit_dim,
            databases,
            dataset_type=dataset,
            coeffs_dir="./noncircular_BBH_fits/Fitting_coefficients",
        )
    except FileNotFoundError:
        coeffs = read_fit_coefficients(
            quantity_to_fit,
            fitting_quantities_string,
            template_model,
            fit_dim,
            databases,
            dataset_type=dataset,
            coeffs_dir="./ringdown/noncircular_BBH_fits/Fitting_coefficients",
        )

    result = template(coeffs, fitting_qs_d, template_model)
    return result


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # example
    nu = 0.25
    chieff = 0.0
    Hefft = np.linspace(0.86, 0.96)  # 1 + (Et**2 - 1.)/(2*nu)
    bmrg = np.linspace(1.8, 3.4)
    jmrg = np.linspace(1.8, 3.4)

    # make sure that we can reproduce greg's plots
    fitting_quantities_dict = {
        "Heff_til": Hefft,
        "nu": nu * np.ones(len(Hefft)),
        "chieff": chieff * np.ones(len(Hefft)),
        "b_massless_EOB": bmrg,
        "Jmrg_til": jmrg,
    }

    # Final mass
    # S=0 Q=1
    qnts = select_fitting_quantities("non-spinning-equal-mass", "Mf")[0]
    result = eval_fit(
        "Mf",
        fitting_quantities_dict,
        qnts,
        "non-spinning-equal-mass",
        databases=["RIT", "SXS", "ET"],
        verbose=True,
    )
    # S=0 Q neq 1
    qnts = select_fitting_quantities("non-spinning", "Mf")[0]
    result2 = eval_fit(
        "Mf",
        fitting_quantities_dict,
        qnts,
        "non-spinning",
        databases=["RIT", "SXS", "ET"],
        verbose=True,
    )
    # S neq 0 Q=1
    qnts = select_fitting_quantities("aligned-spins-equal-mass", "Mf")[0]
    result3 = eval_fit(
        "Mf",
        fitting_quantities_dict,
        qnts,
        "aligned-spins-equal-mass",
        databases=["RIT"],
        verbose=True,
    )
    plt.plot(Hefft, result, label="non-spinning-equal-mass")
    plt.plot(Hefft, result2, label="non-spinning")
    plt.plot(Hefft, result3, label="aligned-spins-equal-mass")
    plt.xlabel("Heff_til")
    plt.ylabel("Mf")
    plt.legend()
    plt.show()

    # Final spin
    # S=0 Q=1
    qnts = select_fitting_quantities("non-spinning-equal-mass", "af")[0]
    result = eval_fit(
        "af",
        fitting_quantities_dict,
        qnts,
        "non-spinning-equal-mass",
        databases=["RIT", "SXS", "ET"],
        verbose=True,
    )
    # S=0 Q neq 1
    qnts = select_fitting_quantities("non-spinning", "af")[0]
    result2 = eval_fit(
        "af",
        fitting_quantities_dict,
        qnts,
        "non-spinning",
        databases=["RIT", "SXS", "ET"],
        verbose=True,
    )
    # S neq 0 Q=1
    qnts = select_fitting_quantities("aligned-spins-equal-mass", "af")[0]
    result3 = eval_fit(
        "af",
        fitting_quantities_dict,
        qnts,
        "aligned-spins-equal-mass",
        databases=["RIT"],
        verbose=True,
    )
    plt.plot(jmrg, result, label="non-spinning-equal-mass")
    plt.plot(jmrg, result2, label="non-spinning")
    plt.plot(jmrg, result3, label="aligned-spins-equal-mass")
    plt.xlabel("J_mrg_til")
    plt.ylabel("af")
    plt.legend()
    plt.show()

    # Peak frequency
    # S=0 Q=1
    qnts = select_fitting_quantities("non-spinning-equal-mass", "omega_peak22")[0]
    result = eval_fit(
        "omega_peak22",
        fitting_quantities_dict,
        qnts,
        "non-spinning-equal-mass",
        databases=["RIT", "SXS", "ET"],
        verbose=True,
    )
    # S=0 Q neq 1
    qnts = select_fitting_quantities("non-spinning", "omega_peak22")[0]
    result2 = eval_fit(
        "omega_peak22",
        fitting_quantities_dict,
        qnts,
        "non-spinning",
        databases=["RIT", "SXS", "ET", "RWZ"],
        verbose=True,
    )
    # S neq 0 Q=1
    qnts = select_fitting_quantities("aligned-spins-equal-mass", "omega_peak22")[0]
    result3 = eval_fit(
        "omega_peak22",
        fitting_quantities_dict,
        qnts,
        "aligned-spins-equal-mass",
        databases=["RIT"],
        verbose=True,
    )
    plt.plot(bmrg, result, label="non-spinning-equal-mass")
    plt.plot(bmrg, result2, label="non-spinning")
    plt.plot(bmrg, result3, label="aligned-spins-equal-mass")
    plt.xlabel("b_massless_EOB")
    plt.ylabel("omega_peak22")
    plt.legend()
    plt.show()

    # Peak amplitude
    # S=0 Q=1
    qnts = select_fitting_quantities("non-spinning-equal-mass", "A_peak22")[0]
    result = eval_fit(
        "A_peak22",
        fitting_quantities_dict,
        qnts,
        "non-spinning-equal-mass",
        databases=["RIT", "SXS", "ET"],
        verbose=True,
    )
    # S=0 Q neq 1
    qnts = select_fitting_quantities("non-spinning", "A_peak22")[0]
    result2 = eval_fit(
        "A_peak22",
        fitting_quantities_dict,
        qnts,
        "non-spinning",
        databases=["RIT", "SXS", "ET", "RWZ"],
        verbose=True,
    )
    # S neq 0 Q=1
    qnts = select_fitting_quantities("aligned-spins-equal-mass", "A_peak22")[0]
    result3 = eval_fit(
        "A_peak22",
        fitting_quantities_dict,
        qnts,
        "aligned-spins-equal-mass",
        databases=["RIT"],
        verbose=True,
    )
    plt.plot(bmrg, result, label="non-spinning-equal-mass")
    plt.plot(bmrg, result2, label="non-spinning")
    plt.plot(bmrg, result3, label="aligned-spins-equal-mass")
    plt.xlabel("b_massless_EOB")
    plt.ylabel("A_peak22")
    plt.legend()
    plt.show()
