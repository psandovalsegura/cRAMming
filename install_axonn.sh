#!/bin/bash

# install AxoNN
git clone git@github.com:axonn-ai/axonn.git
cd axonn
git checkout new-easy-api
pip install -e .

cd ..
