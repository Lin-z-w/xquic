#!/bin/bash

if [ -d "build" ]; then
    sudo rm -r build/ 
fi

SSL_TYPE_STR="boringssl"
SSL_PATH_STR="/home/cnic/lzw/xquic/third_party/boringssl"

# build XQUIC with BoringSSL
# When build XQUIC with boringssl, by default XQUIC will use boringssl
# in third_party. If boringssl is deployed in other directories, SSL_PATH could be 
# used to specify the search path of boringssl
git submodule update --init --recursive
mkdir -p build; cd build
cmake -DGCOV=off -DCMAKE_BUILD_TYPE=Debug -DXQC_ENABLE_TESTING=1 -DXQC_SUPPORT_SENDMMSG_BUILD=1 -DXQC_ENABLE_EVENT_LOG=0 -DXQC_ONLY_ERROR_LOG=1 -DXQC_ENABLE_BBR2=1 -DXQC_ENABLE_RENO=1 -DXQC_ENABLE_UNLIMITED=1 -DXQC_PRINT_SECRET=1 -DSSL_TYPE=${SSL_TYPE_STR} -DSSL_PATH=${SSL_PATH_STR} ..

# exit if cmake error
if [ $? -ne 0 ]; then
    echo "cmake failed"
    exit 1
fi

make -j
