#pragma once

#include "../../detail/prologue.hpp"
#include <concepts>
#include <type_traits>
#include <utility>

namespace ubu
{
namespace detail
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

} // end detail


// to_integral converts a type into an integral
// XXX to_integral should be a CPO
template<class T>
  requires (std::integral<T> or detail::convertible_to_integral_value_type<T>)
constexpr std::integral auto to_integral(const T& i)
{
  if constexpr (std::integral<T>)
  {
    // case 0: T is already std::integral
    return i;
  }
  else
  {
    // case 1: T is convertible to std::integral value_type
    using value_type = typename std::remove_cvref_t<T>::value_type;
    return static_cast<value_type>(i);
  }
}


template<class T>
using to_integral_t = decltype(to_integral(std::declval<T>()));


} // end ubu

#include "../../detail/epilogue.hpp"

