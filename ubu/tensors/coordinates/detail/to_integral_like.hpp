#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/integrals/integral_like.hpp"
#include "../../../utilities/tuples.hpp"
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ubu::detail
{

template<class T>
concept single_integral_like = 
  tuples::single_like<T>
  and integral_like<tuples::first_t<T>>
;

template<class C>
  requires (integral_like<std::remove_cvref_t<C>> or single_integral_like<C>)
constexpr decltype(auto) to_integral_like(C&& coord)
{
  if constexpr(tuples::tuple_like<C>)
  {
    return get<0>(std::forward<C>(coord));
  }
  else
  {
    return std::forward<C>(coord);
  }
}

template<class C>
using to_integral_like_t = std::remove_cvref_t<decltype(to_integral_like(std::declval<C>()))>;

} // end ubu::detail

#include "../../../detail/epilogue.hpp"

