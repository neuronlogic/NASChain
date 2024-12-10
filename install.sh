#!/bin/bash

PYTHON_EXEC=python

echo "Installing primary packages..."

# Install the main packages
$PYTHON_EXEC -m pip install pymoo==0.3.0
$PYTHON_EXEC -m pip install torch==2.2.0
$PYTHON_EXEC -m pip install torchvision==0.17.0
$PYTHON_EXEC -m pip install pandas==2.2.3
$PYTHON_EXEC -m pip install pynvml==11.5.0
$PYTHON_EXEC -m pip install huggingface-hub==0.23.2
$PYTHON_EXEC -m pip install calflops==0.2.9
$PYTHON_EXEC -m pip install transformers==4.41.2
$PYTHON_EXEC -m pip install plotly==5.20.0
$PYTHON_EXEC -m pip install wandb==0.19.0
$PYTHON_EXEC -m pip install onnx-tool==0.9.0
$PYTHON_EXEC -m pip install expecttest==0.1.6
$PYTHON_EXEC -m pip install loguru

echo "Installing additional dependencies for bittensor..."

# Install additional dependencies required for bittensor
$PYTHON_EXEC -m pip install aiohttp~=3.9
$PYTHON_EXEC -m pip install async-property==0.2.2
$PYTHON_EXEC -m pip install bittensor-cli
$PYTHON_EXEC -m pip install bittensor-wallet>=2.1.3
$PYTHON_EXEC -m pip install bt-decode==0.4.0
$PYTHON_EXEC -m pip install colorama~=0.4.6
$PYTHON_EXEC -m pip install fastapi~=0.110.1
$PYTHON_EXEC -m pip install msgpack-numpy-opentensor~=0.5.0
$PYTHON_EXEC -m pip install munch~=2.5.0
$PYTHON_EXEC -m pip install nest-asyncio
$PYTHON_EXEC -m pip install netaddr
$PYTHON_EXEC -m pip install pycryptodome<4.0.0,>=3.18.0
$PYTHON_EXEC -m pip install pydantic<3,>=2.3
$PYTHON_EXEC -m pip install python-Levenshtein
$PYTHON_EXEC -m pip install python-statemachine~=2.1
$PYTHON_EXEC -m pip install retry
$PYTHON_EXEC -m pip install rich
$PYTHON_EXEC -m pip install scalecodec==1.2.11
$PYTHON_EXEC -m pip install substrate-interface~=1.7.9
$PYTHON_EXEC -m pip install uvicorn
$PYTHON_EXEC -m pip install websockets>=14.1

$PYTHON_EXEC -m pip install bittensor==8.4.5 --no-deps
$PYTHON_EXEC -m pip install numpy==1.23.5

echo "Downgrading setuptools to match bittensor's requirement..."
$PYTHON_EXEC -m pip install setuptools~=70.0.0

echo "All packages installed successfully!"
