import click
import minterpy as mp
import numpy as np
import pickle

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from typing import Any, Callable

from test_functions import (
    runge,
    f2,
    f3,
    multidim_sin,
    multidim_cos,
    multidim_cos_sin,
)

TEST_FUNCTIONS = {
    "runge": runge,
    "f2": f2,
    "f3": f3,
    "f4": multidim_cos_sin,
    "multidim_sin": multidim_sin,
    "multidim_cos": multidim_cos,
}

SEED_NUMBER = 1228457


def create_interpolant(
    poly_degree: int,
    my_func: Callable,
    num_dim: int,
    lp_degree: float,
    parameter: Any,
):
    """Create an interpolant given a Callable."""

    # --- Create a Minterpy interpolant
    if parameter:
        if len(parameter) == 1:
            my_fun = partial(my_func, parameter=parameter[0])
        else:
            my_fun = partial(my_func, parameter=parameter)
    else:
        my_fun = my_func

    interp_minterpy = mp.interpolate(
        my_fun, int(num_dim), int(poly_degree), lp_degree
    )

    # NOTE: interpolant object cannot be pickled; use Newton
    return interp_minterpy.interpolator(my_fun)


@click.command()
@click.option(
    "-fn",
    "--function-name",
    required=True,
    type=click.Choice(list(TEST_FUNCTIONS.keys()), case_sensitive=False),
    help="Name of the test function",
)
@click.option(
    "-m",
    "--num-dim",
    required=True,
    type=int,
    help="Number of spatial dimensions",
)
@click.option(
    "-n",
    "--poly-degrees",
    nargs=2,
    required=True,
    type=int,
    help="Range of polynomial degrees",
)
@click.option(
    "-p",
    "--lp-degree",
    required=True,
    type=str,
    help="lp-degree of the polynomial",
)
@click.option(
    "-p",
    "--param",
    multiple=True,
    show_default=True,
    help="Parameter for the test function",
)
def run_minterpy_interpolant(
    function_name,
    num_dim,
    poly_degrees,
    lp_degree,
    param,
):

    # ---
    param_values = [float(p) for p in param]
    poly_degree_start, poly_degree_end = poly_degrees
    test_function = TEST_FUNCTIONS[function_name]

    # String to save the result
    if lp_degree == "inf":
        lp_degree_str = "inf"
    else:
        lp_degree_str = int(float(lp_degree))
    
    if param:
        param_str = ["-param"] + [str(p) for p in param]
        param_str = "_".join(param_str)
    else:
        param_str = "-param_default"


    # --- Parallel processing
    poly_degrees_range = np.arange(poly_degree_start, poly_degree_end + 1)

    func = partial(
        create_interpolant,
        my_func=test_function,
        num_dim=num_dim,
        lp_degree=float(lp_degree),
        parameter=param_values,
    )

    pool = Pool(processes=10)
    for idx, res in enumerate(tqdm(pool.imap(func, poly_degrees_range), total=len(poly_degrees_range))):

        # Save the resulting Newton polynomial object
        fname = (
            f"interpolant-{function_name}-{num_dim}-{lp_degree_str}"
            f"{param_str}-{poly_degrees_range[idx]:03}.pkl"
        )
        with open(fname, "wb") as f:
            pickle.dump(res, f)


if __name__ == "__main__":
    run_minterpy_interpolant()
