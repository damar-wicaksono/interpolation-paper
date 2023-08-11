import click
import minterpy as mp
import numpy as np

from functools import partial
from multiprocessing import Pool
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
    "multidim_sin": multidim_sin,
    "multidim_cos": multidim_cos,
    "multidim_cos_sin": multidim_cos_sin,
}

SEED_NUMBER = 1228457

def compute_interpolation_error(
    poly_degree: int,
    my_func: Callable,
    num_dim: int,
    lp_degree: float,
    xx_test: np.ndarray,
    yy_test: np.ndarray,
    num_batches: int,
    parameter: Any,
):
    """Compute the interpolation L1 error of a given test function.
    
    Notes
    -----
    - The parameter `poly_degree` must appear first because
      the paralellization is with respect to that.
    """

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
    
    # --- Compute batch size
    num_test_points = yy_test.shape[0]
    batch_size = int(np.ceil(num_test_points / num_batches))

    # --- Compute the L1 error
    errors_temp = np.zeros(yy_test.shape)

    for i in range(num_batches):
        idx_1 = i * batch_size
        if i == num_batches - 1:
            idx_2 = num_test_points
        else:
            idx_2 = (i + 1) * batch_size
        errors_temp[idx_1:idx_2] = (
            np.abs(yy_test[idx_1:idx_2]
            - interp_minterpy(xx_test[idx_1:idx_2]))
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
def run_minterpy_benchmark(
    function_name,
    num_dim,
    poly_degrees,
    lp_degree,
    test_sample_size,
    num_batches,
    param,
):

    # ---
    param_values = [float(p) for p in param]
    poly_degree_start, poly_degree_end = poly_degrees
    test_function = TEST_FUNCTIONS[function_name]

    # --- Create a testing dataset
    rng = np.random.default_rng(SEED_NUMBER)
    xx_test = -1 + 2 * rng.random((test_sample_size, num_dim))
    yy_test = test_function(xx_test)

    # --- Set the arguments of fun
    func = partial(
        compute_interpolation_error,
        my_func=test_function,
        num_dim=num_dim,
        lp_degree=float(lp_degree),
        xx_test=xx_test,
        yy_test=yy_test,
        num_batches=num_batches,
        parameter=param_values,
    )

    # --- Parallel processing
    poly_degrees_range = np.arange(poly_degree_start, poly_degree_end + 1)
    errors_minterpy = np.zeros(poly_degrees_range.shape)

    pool = Pool()
    for ind, res in enumerate(pool.imap(func, poly_degrees_range)):
        errors_minterpy[ind] = res

    # Save the result
    if lp_degree == "inf":
        lp_degree_str = "inf"
    else:
        lp_degree_str = int(float(lp_degree))
    
    if param:
        param_str = ["-param"] + [str(p) for p in param]
        param_str = "-".join(param_str)
    else:
        param_str = ""

    poly_degrees_str = "-".join([str(n) for n in poly_degrees])
 
    np.savetxt(
        f"errors-{function_name}-{num_dim}-{poly_degrees_str}-{lp_degree_str}{param_str}.csv",
        errors_minterpy,
        delimiter=",",
    )

    # --- Compute the number of points
    num_coeffs = np.zeros(poly_degrees_range.shape)
    for i, poly_degree in enumerate(poly_degrees_range):
        num_coeffs[i] = len(mp.MultiIndexSet.from_degree(2, int(poly_degree), float(lp_degree)))

    np.savetxt(
        f"num-coeffs-{function_name}-{num_dim}-{poly_degrees_str}-{lp_degree_str}{param_str}.csv",
        errors_minterpy,
        delimiter=",",
    )


if __name__ == "__main__":
    run_minterpy_benchmark()
