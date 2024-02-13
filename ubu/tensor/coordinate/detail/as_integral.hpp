#pragma once

#include "../../../detail/prologue.hpp"

#include "../detail/tuple_algorithm.hpp"
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ubu::detail
{

template<class T>
concept single_integral = 
  tuple_like_of_size<T,1>
  and std::integral<std::remove_cvref_t<std::tuple_element_t<0,std::remove_cvref_t<T>>>>
;

template<class C>
  requires (std::integral<std::remove_cvref_t<C>> or single_integral<C>)
constexpr decltype(auto) as_integral(C&& coord)
{
  if constexpr(tuple_like<C>)
  {
    return get<0>(std::forward<C>(coord));
  }
  else
  {
    return std::forward<C>(coord);
  }
}

template<class C>
using as_integral_t = std::remove_cvref_t<decltype(as_integral(std::declval<C>()))>;

} // end ubu::detail

#include "../../../detail/epilogue.hpp"

