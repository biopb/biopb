#! /bin/bash

# generate descriptor file
protoc -o gen/descriptor.pb --include_source_info biopb/*/*

# generate python files
python -m grpc_tools.protoc -I . --python_out=gen/python/ --grpc_python_out=gen/python/ biopb/image/*

# generate java files
# protoc --plugin=protoc-gen-grpc-java --grpc-java_out=gen/java/ --proto_path=. biopb/*/*
