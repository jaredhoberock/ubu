#pragma once

#include "../../../detail/prologue.hpp"
#include "../../../miscellaneous/constant_valued.hpp"
#include <concepts>
#include <type_traits>

namespace ubu::detail
{

template<class T>
concept convertible_to_integral_value_type =
  requires()
  {
    // ::value_type must exist
    typename std::remove_cvref_t<T>::value_type;
  }

  // ::value_type must be an integral type
  and std::integral<typename std::remove_cvref_t<T>::value_type>

  // T must be convertible to ::value_type
  and std::convertible_to<T, typename std::remove_cvref_t<T>::value_type>
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
    // case 1: I is convertible to std::integral value_type
    using value_type = typename std::remove_cvref_t<I>::value_type;
    return static_cast<value_type>(i);
  }
}

} // end ubu::detail

#include "../../../detail/epilogue.hpp"

