#pragma once

#include "../../detail/prologue.hpp"
#include <iostream>
#include <type_traits>

namespace ubu
{
namespace detail
{

struct underscore_t {};

std::ostream& operator<<(std::ostream& os, underscore_t)
{
  return os << "_";
}

template<class T>
constexpr bool is_underscore_v = std::is_same_v<std::remove_cvref_t<T>,underscore_t>;

} // end detail

constexpr detail::underscore_t _;

} // end ubu

#if __has_include(<fmt/core.h>)
#include <fmt/core.h>

namespace fmt
{

template<>
struct formatter<ubu::detail::underscore_t>
{
  constexpr auto parse(format_parse_context& ctx)
  {
    return ctx.begin();
  }

  template<class FormatContext>
  auto format(ubu::detail::underscore_t, FormatContext& ctx)
  {
    return format_to(ctx.out(), "_");
  }
};

} // end fmt

#endif // libfmt

#include "../../detail/epilogue.hpp"

