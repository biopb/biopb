cmake_minimum_required(VERSION 3.16)

project(BioPB C CXX)

set(Protobuf_INCLUDE_DIRS, "../biopb/")

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${Protobuf_INCLUDE_DIRS})

option(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")

find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

find_package(Threads)

add_library(biopb
    SHARED 
    "../biopb/image/bindata.proto"
    "../biopb/image/detection_request.proto"
    "../biopb/image/detection_response.proto"
    "../biopb/image/detection_settings.proto"
    "../biopb/image/image_data.proto"
    "../biopb/image/roi.proto"
    "../biopb/image/rpc_object_detection.proto"
    "../biopb/image/rpc_process_image.proto")

target_link_libraries(biopb PUBLIC gRPC::grpc++)

protobuf_generate(
	TARGET biopb
	LANGUAGE cpp
	IMPORT_DIRS "../"
	OUT_VAR proto_gen_files)
