#pragma once

#include "../../../detail/prologue.hpp"

#include "../concepts/coordinate.hpp"
#include <type_traits>
#include <utility>

namespace ubu::detail
{

template<scalar_coordinate C>
constexpr decltype(auto) as_integral(C&& coord)
{
  if constexpr (std::integral<std::remove_cvref_t<C>>) return std::forward<C>(coord);
  else return get<0>(std::forward<C>(coord));
}

template<scalar_coordinate C>
using as_integral_t = std::remove_cvref_t<decltype(as_integral(std::declval<C>()))>;

} // end ubu::detail

#include "../../../detail/epilogue.hpp"

