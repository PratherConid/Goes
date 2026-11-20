#pragma once
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Mirrors JS's Number.isInteger().
inline bool is_integer(double x) { return std::isfinite(x) && x == std::floor(x); }

// Stands in for JS's Number->string (`${x}`/String(x)): an integral value prints with no decimal
// point, matching JS; otherwise 15 significant digits.
inline std::string js_number_string(double d) {
    if (std::isnan(d)) return "NaN";
    if (std::isinf(d)) return d > 0 ? "Infinity" : "-Infinity";
    if (d == std::floor(d) && std::abs(d) < 1e15) return std::to_string(static_cast<long long>(d));
    std::ostringstream oss;
    oss.precision(15);
    oss << d;
    return oss.str();
}

namespace always_assert_detail {
template <class T>
void append(std::ostringstream& out, const T& part) { out << part; }
// JS's `${x}` for a number.
inline void append(std::ostringstream& out, double part) { out << js_number_string(part); }
// JS's `${xs}` (Array.prototype.toString()): comma-joined, no spaces.
template <class T>
void append(std::ostringstream& out, const std::vector<T>& part) {
    for (std::size_t i = 0; i < part.size(); i++) { if (i) out << ','; out << part[i]; }
}
template <class... MsgParts>
[[noreturn]] void fail(const MsgParts&... msg_parts) {
    std::ostringstream out;
    out << "Assertion failed: ";
    (append(out, msg_parts), ...);
    throw std::runtime_error(out.str());
}
}

// Mirrors shared/types.ts's assert(), throwing std::runtime_error("Assertion failed: " + msg) -
// active in every build, unlike <cassert>'s assert() (compiled out by Release's NDEBUG). msg is
// `msg_parts` streamed together, so a TS template literal's interpolated values become separate
// parts, formatted only on failure.
template <class... MsgParts>
inline void always_assert(bool cond, const MsgParts&... msg_parts) {
    // Decayed, so string literals of different lengths share one instantiation of fail().
    if (!cond) [[unlikely]] always_assert_detail::fail<std::decay_t<const MsgParts>...>(msg_parts...);
}
