#! /bin/sh

protoc -o ./descriptor.pb --include_source_info ../biopb/*/*

sabledocs