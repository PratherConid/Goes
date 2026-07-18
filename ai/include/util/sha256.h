#pragma once
#include <string>

// SHA-256 digest of `input`, as 64 lowercase hex characters. Self-contained
// (no external dependency) - httplib.h's bundled hash is SHA-1 (20 bytes/40
// hex chars), and its SHA-256 support is behind CPPHTTPLIB_OPENSSL_SUPPORT/
// mbedtls/wolfssl, none of which this project enables (see server.cpp's
// "OpenSSL is opt-in" comment).
std::string sha256_hex(const std::string& input);
