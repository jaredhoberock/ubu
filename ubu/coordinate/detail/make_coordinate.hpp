#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate.hpp"
#include "../rank.hpp"
#include <concepts>
#include <cstdint>
#include <utility>


// the reason that make_coordinate exists is to provide a uniform means of constructing
// a coordinate, which is otherwise defied by std::array's weird construction syntax


namespace ubu::detail
{


template<coordinate T>
struct make_coordinate_impl
{
  template<class... Args>
    requires std::constructible_from<T,Args&&...>
  constexpr T operator()(Args&&... args) const
  {
    return T{std::forward<Args>(args)...};
  }
};


// specialization for std::array, which requires the weird doubly-nested brace syntax
template<class T, std::size_t n>
struct make_coordinate_impl<std::array<T,n>>
{
  template<class... Args>
  constexpr std::array<T,n> operator()(Args&&... args) const
  {
    return std::array<T,n>{{std::forward<Args>(args)...}};
  }
};


// scalar case
template<number T, class Arg>
  requires std::constructible_from<T,Arg&&>
constexpr T make_coordinate(Arg&& arg)
{
  return {std::forward<Arg>(arg)};
}


// non-scalar case
template<coordinate T, class... Args>
  requires (not std::integral<T> and rank_v<T> == sizeof...(Args))
constexpr T make_coordinate(Args&&... args)
{
  detail::make_coordinate_impl<T> impl;
  return impl(std::forward<Args>(args)...);
}


} // end ubu::detail


#include "../../detail/epilogue.hpp"

