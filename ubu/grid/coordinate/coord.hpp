#pragma once

#include "../../detail/prologue.hpp"
#include "coordinate.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

template<class T>
concept has_coord_member_variable = requires(T arg)
{
#if defined(__circle_lang__)
  arg.coord;
  requires requires(decltype(arg.coord) c)
  {
    { c } -> coordinate;
  };
#else
  { arg.coord } -> coordinate;
#endif
};

template<class T>
concept has_coord_member_function = requires(T arg)
{
  { arg.coord() } -> coordinate;
};

template<class T>
concept has_coord_free_function = requires(T arg)
{
  { coord(arg) } -> coordinate;
};

struct dispatch_coord
{
  template<class T>
    requires has_coord_member_variable<T&&>
  constexpr auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).coord;
  }

  template<class T>
    requires (not has_coord_member_variable<T&&>
              and has_coord_member_function<T&&>)
  constexpr auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).coord();
  }

  template<class T>
    requires (not has_coord_member_variable<T&&>
              and not has_coord_member_function<T&&>
              and has_coord_free_function<T&&>)
  constexpr auto operator()(T&& arg) const
  {
    return coord(std::forward<T>(arg));
  }
};

} // end detail


constexpr detail::dispatch_coord coord;

} // end ubu

#include "../../detail/epilogue.hpp"

