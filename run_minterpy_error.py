import click
import minterpy as mp
import numpy as np
import pickle

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from typing import Callable

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
INTERPOLANTS_LOCATION = "./results/interpolants/"

def compute_interpolation_error(
    idx_batch: int,
    minterpy_interpolant: Callable,
    xx_test: np.ndarray,
    yy_test: np.ndarray,
    num_batches: int,
):
    """Compute the interpolation L1 error of a given test function.
    
    Notes
    -----
    - The parameter `poly_degree` must appear first because
      the paralellization is with respect to that.
    """
    # --- Compute batch size
    num_test_points = yy_test.shape[0]
    batch_size = int(np.ceil(num_test_points / num_batches))

    # --- Compute the L1 error
    idx_1 = idx_batch * batch_size
    if idx_batch == num_batches - 1:
        idx_2 = num_test_points
    else:
        idx_2 = (idx_batch + 1) * batch_size
    errors_temp = (
        np.abs(yy_test[idx_1:idx_2]
        - minterpy_interpolant(xx_test[idx_1:idx_2]))
    )

    return np.max(errors_temp)


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
    "-s",
    "--test-sample-size",
    default=1000000,
    show_default=True,
    type=int,
    help="Number of test sample points",
)
@click.option(
    "-b",
    "--num-batches",
    default=1000,
    show_default=True,
    type=int,
    help="Number of batches for test point evaluations."
)
@click.option(
    "-p",
    "--param",
    multiple=True,
    show_default=True,
    help="Parameter for the test function",
)
def run_minterpy_error(
    function_name,
    num_dim,
    poly_degrees,
    lp_degree,
    test_sample_size,
    num_batches,
    param,
):

    # ---
    poly_degree_start, poly_degree_end = poly_degrees
    test_function = TEST_FUNCTIONS[function_name]
    param_values = [float(p) for p in param]

    # --- Create a testing dataset
    rng = np.random.default_rng(SEED_NUMBER)
    xx_test = -1 + 2 * rng.random((test_sample_size, num_dim))
    if param_values:
        if len(param) == 1:
            yy_test = test_function(xx_test, parameter=param_values[0])
        else:
            yy_test = test_function(xx_test, parameter=param_values)
    else:
        yy_test = test_function(xx_test)

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
    errors_minterpy = np.zeros(poly_degrees_range.shape)

    for poly_degree in poly_degrees_range:
        
        # Save the resulting Newton polynomial object
        dirname = (
            f"{INTERPOLANTS_LOCATION}/{function_name}/dim-{num_dim}"
            f"/lp-{lp_degree_str}/{param_str.replace('-', '').replace('_', '-')}/"
        )
        fname = (
            f"{dirname}interpolant-{function_name}-{num_dim}"
            f"-{lp_degree_str}{param_str}-{poly_degree:03}.pkl"
        )
        with open(fname, "rb") as f:
            minterpy_interpolant = pickle.load(f)
    
        func = partial(
            compute_interpolation_error,
            minterpy_interpolant=minterpy_interpolant,
            xx_test=xx_test,
            yy_test=yy_test,
            num_batches=num_batches,
        )

        errors_minterpy_temp = np.zeros(num_batches)

        pool = Pool(processes=6)
        for ind, res in enumerate(tqdm(pool.imap(func, np.arange(num_batches)), total=num_batches)):
            errors_minterpy_temp[ind] = res

        errors_minterpy = np.max(errors_minterpy_temp)
    
        np.savetxt(
            f"errors-{function_name}-{num_dim}-{lp_degree_str}{param_str}-{int(poly_degree):03}.csv",
            [errors_minterpy],
            delimiter=",",
        )

        # --- Compute the number of points
        num_coeffs = len(mp.MultiIndexSet.from_degree(int(num_dim), int(poly_degree), float(lp_degree)))
        
        np.savetxt(
            f"num-coeffs-{function_name}-{num_dim}-{lp_degree_str}-{int(poly_degree):03}.csv",
            [num_coeffs],
            delimiter=",",
        )


if __name__ == "__main__":
    run_minterpy_error()
