#pragma once

#include "../detail/prologue.hpp"

#if (not __has_include(<fmt/format-inl.h>) or not __has_include(<fmt/compile.h>))
#error "The print.hpp header requires fmtlib."
#endif

#include <cassert>
#include <cstdio>
#include <exception>

#if __has_include(<nv/target>)
#include <nv/target>
#endif // __has_include

// ubu::print is a wrapper around fmtlib, which needs to be configured in a certain way to work in GPU code 

namespace ubu::detail
{


// this function will be used if fmtlib fails an assert
inline void fmt_assert_fail(const char* file, int line, const char* message, const char* function) noexcept
{
#if defined(__circle_lang__) and defined(__CUDACC__) and __has_include(<nv/target>)
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (
    // GPU code has no access to either std::fprintf or std::terminate
    __assert_fail(message, file, line, function);
  ), (
    // Use unchecked std::fprintf to avoid triggering another assertion when
    // writing to stderr fails
    std::fprintf(stderr, "%s:%d: assertion failed: %s", file, line, message);
    // Chosen instead of std::abort to satisfy Clang in CUDA mode during device
    // code pass.
    std::terminate();
  ))
#else
  // Use unchecked std::fprintf to avoid triggering another assertion when
  // writing to stderr fails
  std::fprintf(stderr, "%s:%d: assertion failed: %s", file, line, message);
  // Chosen instead of std::abort to satisfy Clang in CUDA mode during device
  // code pass.
  std::terminate();
#endif
}


} // end ubu::detail

// configure fmtlib
// note that this configuration will interfere with the user's configuration of fmtlib, if any
#define FMT_HEADER_ONLY
#define FMT_USE_INT128 0 // XXX This works around circle's LLVM ERROR: Undefined external symbol "__udivti3"
#define FMT_STATIC_THOUSANDS_SEPARATOR ',' // this avoids using the <locale> header, which causes problems with print in GPU code
#define FMT_ASSERT(condition, message) ((condition) ? (void) 0 : ::ubu::detail::fmt_assert_fail(__FILE__, __LINE__, (message), __FUNCTION__))
#define FMT_USE_NONTYPE_TEMPLATE_ARGS 1 // make operator""_cf available

// include fmtlib
#include <fmt/format-inl.h>
#include <fmt/compile.h>

namespace ubu
{
namespace detail
{


// detail::format is just a wrapper around some lower-level fmtlib functions
// we can't use either fmt::format or std::format because their use introduces
// non-trivial LLVM module constructors, which NVPTX does not support
template<class S, class... Args>
constexpr std::string format(const S& fmt_str, Args&&... args)
{
  auto sz = fmt::formatted_size(fmt_str, args...);
  std::string result(sz, 0);
  fmt::format_to(result.data(), fmt_str, args...);
  return result;
}


} // detail


inline namespace literals
{

// use libfmt's user-defined compiled formatting string literal operator
using fmt::operator""_cf;

} // end literals


// ubu::print is a wrapper around fmtlib
// The reason ubu::print exists is because fmt::print is insufficient. It uses fwrite, which is unavailable in device code.
//
// The fmt_str parameter must be a "compiled" formatting string created with FMT_COMPILE("my formatting string...") or "my formatting string..."_cf
//
// For example,
//
//     using namespace ubu::literals;
//
//     ubu::print("Hello! Here is some formatted output: {}, {}, {}\n"_cf, 1, 2, "string");
//
//     // The following is printed to the terminal:
//     // Hello! Here is some formatted output: 1, 2, string
//
template<class S, class... Args>
constexpr void print(const S& fmt_str, Args&&... args)
{
  static_assert(fmt::detail::is_compiled_string<S>::value,
    "ubu::print requires a compiled format string (did you forget to append suffix _cf ?).");

  // ubu::print simply calls printf on the result of detail::format
  printf("%s", detail::format(fmt_str, args...).c_str());
}

template<class S, class... Args>
constexpr void println(const S& fmt_str, Args&&... args)
{
  static_assert(fmt::detail::is_compiled_string<S>::value,
    "ubu::print requires a compiled format string (did you forget to append suffix _cf ?).");

  // ubu::println simply calls printf on the result of detail::format
  printf("%s\n", detail::format(fmt_str, args...).c_str());
}


} // end ubu

#include "../detail/epilogue.hpp"

