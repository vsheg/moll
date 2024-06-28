# Development Environment

## Requirements

- **JAX** is required to run the code.
- **Poetry** is used for dependency management.
- **NVIDIA Driver** and **NVIDIA Container Toolkit** are required for the CUDA version.



## Installation

### CPU Version

To install the CPU-only development environment, run the following command:

```bash
poetry install --with cpu,dev
```

### CUDA Version

1. Ensure the NVIDIA Driver is installed on your machine.
2. Install Docker and NVIDIA Container Toolkit.
3. Run the following command to install the GPU-supported development environment:

```bash
poetry install --with cuda,dev
```

### Manual JAX Installation

To use the local JAX installation, install without specifying additional dependencies:

```bash
poetry install --with dev
```



## Docker Container for Development

Optionally, you can run the development environment in a Docker container. Configuration files are `.devcontainer/*` and `Dockerfile.devcontainer`.

### CPU Version

To install the CPU version, run inside the devcontainer:

```bash
poetry install --with cpu,dev
```

### GPU Version

Install NVIDIA Container Toolkit and run the devcontainer. Then, run the following command:

```bash
poetry install --with cuda,dev
```



## Running Tests

To run the tests, use:

```bash
poetry run pytest
```



## Building Documentation

1. Install documentation-specific dependencies:

```bash
poetry install --with docs
```

2. Build the documentation:

```bash
make docs
```



