#!/bin/bash

# install AxoNN (dependency for SlimAdamW)
git clone git@github.com:axonn-ai/axonn.git
cd axonn
pip install -e .

cd ..

# install SlimAdamW
git clone https://github.com/axonn-ai/SlimAdamW.git
cd SlimAdamW
pip install -e .

cd ..
