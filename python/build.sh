#! /bin/bash

# generate python files
python -m grpc_tools.protoc -I ../ --python_out=. --grpc_python_out=. ../biopb/image/*

cp ../README.md .
cp ../LICENSE .

python -m build
