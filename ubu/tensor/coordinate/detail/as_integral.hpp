#pragma once

#include "../../../detail/prologue.hpp"

#include <concepts>

namespace ubu::detail
{

template<class T>
concept convertible_to_integral_value_type =
  requires(T i)
  {
    // there must be a nested type T::value_type
    typename T::value_type;
  }
  // T::value_type must be std::integral
  and std::integral<typename T::value_type>
  // T must be convertible to T::value_type
  and std::convertible_to<T,typename T::value_type>
;

// as_integral converts a type into an integral
template<class I>
  requires (std::integral<I> or convertible_to_integral_value_type<I>)
constexpr std::integral auto as_integral(const I& i)
{
  if constexpr (std::integral<I>)
  {
    // case 0: I is already std::integral
    return i;
  }
  else
  {
    // case 1: I has a nested value_type and is convertible to it
    return static_cast<typename I::value_type>(i);
  }
}


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

