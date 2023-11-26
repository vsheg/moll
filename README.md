# moll

`moll` is a computational chemistry tool that is currently under development.

## Installation

`moll` depends on JAX, you can install it with CUDA support or without it.

- To install `moll` without CUDA support, run:

    ```bash
    pip install -U moll[cpu]
    ```

- If you have a GPU, it is recommended to install CUDA version of JAX at first. Check [this repo](https://github.com/google/jax) and [docs](https://jax.readthedocs.io/en/latest/installation.html) to see how to install JAX. Then run:

    ```bash
    pip install -U moll
    ```

## How to use

Check [docs](https://vsheg.github.io/moll/) or see examples in `notebooks/` directory in the [repo](https://github.com/vsheg/moll).