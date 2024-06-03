#pragma once

#include "../../../detail/prologue.hpp"
#include <iostream>
#include <type_traits>
#include <cstddef>

namespace ubu
{
namespace detail
{

struct underscore_t
{
  constexpr bool operator==(const underscore_t&) const
  {
    return true;
  }
  
  template<class T>
  constexpr bool operator==(const T&) const
  {
    return false;
  }

  // XXX we can't simply define a static member rank for empty types
  // like underscore_t due to circle's std::tuple bug
  //static constexpr std::size_t rank = 1;
};

// XXX specialize circle_tuple_bug_static_rank_workaround 
//     to WAR circle's std::tuple bug which interferes with ubu::rank_v
template<class T>
struct circle_tuple_bug_static_rank_workaround;

template<>
struct circle_tuple_bug_static_rank_workaround<underscore_t>
{
  static constexpr std::size_t rank = 1;
};


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

#include "../../../detail/epilogue.hpp"

