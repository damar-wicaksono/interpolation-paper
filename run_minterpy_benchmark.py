import click


@click.command()
@click.option(
    "-fn",
    "--function-name",
    required=True,
    type=str,
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
    type=float,
    help="lp-degree of the polynomial",
)
@click.option(
    "-s",
    "--test-sample-size",
    default=100000,
    show_default=True,
    type=int,
    help="Number of test sample points",
)
@click.option(
    "-p",
    "--param",
    multiple=True,
    show_default=True,
    help="Parameter for the test function",
)
def run_minterpy_benchmark(function_name, num_dim, poly_degrees, lp_degree, test_sample_size, param):
    poly_degree_start, poly_degree_end = poly_degrees

    print(function_name)
    print(num_dim)
    print(poly_degree_start, poly_degree_end)
    print(lp_degree)
    print(test_sample_size)
    print(param)

if __name__ == "__main__":
    run_minterpy_benchmark()
