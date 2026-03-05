# XQUIC Development Guide

This document provides guidelines for agents working on the XQUIC codebase.

## Project Overview

XQUIC is a QUIC and HTTP/3 client/server implementation in C, supporting multiple platforms (Linux, macOS, Windows, Android, iOS). Uses CMake for building and CUnit for testing.

## Build Commands

### Basic Build with BoringSSL
```bash
git clone https://github.com/alibaba/xquic.git && cd xquic
git submodule update --init --recursive
git clone https://github.com/google/boringssl.git ./third_party/boringssl
cd ./third_party/boringssl && mkdir -p build && cd build
cmake -DBUILD_SHARED_LIBS=0 -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_CXX_FLAGS="-fPIC" .. && make -j ssl crypto
cd ../..
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DXQC_ENABLE_TESTING=1 -DSSL_TYPE=boringssl -DSSL_PATH=${PWD}/third_party/boringssl .. && make -j
```

### Build Options
- `XQC_ENABLE_TESTING=1` - Enable tests
- `XQC_ENABLE_BBR2=1` - Enable BBRv2
- `XQC_ENABLE_RENO=1` - Enable Reno
- `XQC_ENABLE_FEC=1` - Enable FEC
- `SSL_TYPE` - babassl or boringssl

### Running Tests
```bash
cd build
# All tests
sh ../scripts/xquic_test.sh
# Unit tests only
./tests/run_tests
# Case tests (start server first)
./tests/test_server -l d -e &
sleep 1 && sh ../scripts/case_test.sh
# Single CUnit test: edit tests/unittest/main.c to comment out unwanted tests, then rebuild
make run_tests && ./tests/run_tests
```

## Code Style (Nginx-based)

### Formatting
- **Max line width:** 80 characters
- **Indentation:** 4 spaces (no tabs)
- **No trailing whitespace**
- **No `//` comments** - use `/* */` only

### Naming
- Macros: `xqc_` or `XQC_` prefix
- Types: opaque pointers (e.g., `xqc_stream_t`)
- Functions: `xqc_<module>_<action>`

### Function Definitions
```C
xqc_int_t
xqc_function_name(xqc_arg_t *arg)
{
    /* code */
}
```

### Control Flow
```C
if (condition && another_condition) {
    value = a + b;
}

switch (value) {
case CASE_ONE:
    break;
default:
    break;
}
```

## Type Conventions

Defined in `include/xquic/xquic_typedef.h`:
- `xqc_int_t` - Signed 32-bit
- `xqc_uint_t` - Unsigned 32-bit
- `xqc_bool_t` - Boolean (use `XQC_TRUE`/`XQC_FALSE`)
- `xqc_msec_t`/`xqc_usec_t` - Milliseconds/microseconds (uint64_t)
- `xqc_packet_number_t`/`xqc_stream_id_t` - uint64_t

## Error Handling

- Return `xqc_int_t` (negative for errors)
- Use error codes from `include/xquic/xqc_errno.h`
- Use `XQC_UNLIKELY()` for error paths

```C
if (XQC_UNLIKELY(condition)) {
    xqc_log(log, XQC_LOG_ERROR, "error message");
    return -XQC_ERROR;
}
```

## Import Order
1. System headers (`<stdint.h>`, `<string.h>`)
2. Public headers (`<include/xquic/...>`)
3. Internal headers (`"src/..."`)

## Project Structure
```
xquic/
├── include/xquic/      # Public headers
├── src/
│   ├── common/         # Utilities
│   ├── transport/     # QUIC transport
│   ├── http3/         # HTTP/3 and QPACK
│   ├── tls/           # TLS abstraction
│   └── congestion_control/  # CC algorithms
├── tests/
│   ├── unittest/      # CUnit tests
│   └── case_test.sh   # Integration tests
└── scripts/           # Build/test scripts
```

## Git Conventions
- Branch: `dev/<feature>`, `fix/<name>`, `perf/<item>`, `doc/<name>`
- Commit: `[+]` add, `[-]` remove, `[=]` optimize, `[~]` fix

## Common Issues
- **SSL not found:** Check `SSL_TYPE` and `SSL_PATH`
- **CUnit missing:** `sudo apt-get install libcunit1-dev`
- **Libevent missing:** `sudo apt-get install libevent-dev`
