## Installation

This directory contains files for the c++ binding installation.

### Requirement

Setup [vcpkg](https://vcpkg.io/en/) for you development environment. 

### Steps

Run
```
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake

cmake --build build
```
> **_NOTE:_**  `%VCPKG_ROOT%` is your vcpkg installation location.
