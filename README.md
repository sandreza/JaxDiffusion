# JaxDiffusion
The setup for the repository was as follows
Follow the instructions at https://github.com/astral-sh/uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10 3.11 3.12
uv init jaxuvtest 
cp -r jaxuvtest/* .
rm -r jaxuvtest/

uv venv --python 3.11
source .venv/bin/activate
uv python pin 3.11

uv add jax[cuda12_pip]