# moll

`moll` is a computational chemistry tool that is currently under development.

## Installation

`moll` depends on RDKit and JAX, you need to install them manually.

1. Install RDKit. The easiest way is to use `pip`:

    ```bash
    pip install rdkit
    ```

2. Install JAX. We recommend CUDA version, check [this repo](https://github.com/google/jax) and [documentation](https://jax.readthedocs.io/en/latest/installation.html) to see how to install it. If you don't have a GPU, you can install CPU version:

    ```bash
    pip install -U "jax[cpu]"
    ```

3. Install moll:

    ```bash
    pip install moll
    ```

## How to use

Check [docs](https://vsheg.github.io/moll/) or see examples in `notebooks/` directory in the [repo](https://github.com/vsheg/moll).