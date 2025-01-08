# JaxDiffusion

## Setup Instructions

First change into the `JaxDiffusion` directory and follow the instructions below to set up the repository:

1. Install `uv`:
    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Install Python 3.11 using `uv`:
    ```sh
    uv python install 3.11
    ```

3. Create a virtual environment with Python 3.11 and sync:
    ```sh
    uv venv --python 3.11
    source .venv/bin/activate
    uv sync
    ```
