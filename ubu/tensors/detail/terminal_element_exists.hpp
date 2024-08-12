#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/integrals/integral_like.hpp"
#include "../../utilities/integrals/size.hpp"
#include "../coordinates/comparisons/is_below.hpp"
#include "../coordinates/concepts/congruent.hpp"
#include "../coordinates/concepts/coordinate.hpp"
#include "../shapes/in_domain.hpp"
#include "../shapes/shape.hpp"
#include <concepts>
#include <utility>

namespace ubu::detail
{

template<class T, class C>
concept has_element_exists_member_function = requires(T arg, C coord)
{
  { std::forward<T>(arg).element_exists(std::forward<C>(coord)) } -> std::convertible_to<bool>;
};

template<class T, class C>
concept has_element_exists_free_function = requires(T arg, C coord)
{
  { element_exists(std::forward<T>(arg), std::forward<C>(coord)) } -> std::convertible_to<bool>;
};


template<class T, class C>
concept has_element_exists_customization =
  has_element_exists_member_function<T,C>
  or has_element_exists_free_function<T,C>
;


// this case handles when T has a size and a shape
// and C is congruent with T's shape
template<class T, class C>
concept sized_and_shaped_and_congruent =
  sized<T>
  and shaped<T>
  and congruent<shape_t<T>,C>
;


// this case handles when T may not be fully tensor-like (e.g., invocable)
// and C may not be fully coordinate-like (e.g., general arguments to invocables)
// but the element CPO still works
template<class T, class C>
concept unshaped_with_element_access =
  (not shaped<T>)
  and requires(T obj, C coord)
  {
    element(std::forward<T>(obj), std::forward<C>(coord));
  }
;


template<class T, class C>
concept has_terminal_element_exists =
  has_element_exists_customization<T,C>
  or sized_and_shaped_and_congruent<T,C>
  or unshaped_with_element_access<T,C>;
;


template<class T, class C>
  requires has_terminal_element_exists<T&&,C&&>
constexpr bool terminal_element_exists(T&& obj, C&& coord)
{
  if constexpr (has_element_exists_member_function<T&&,C&&>)
  {
    return std::forward<T>(obj).element_exists(std::forward<C>(coord));
  }
  else if constexpr (has_element_exists_free_function<T&&,C&&>)
  {
    return element_exists(std::forward<T>(obj), std::forward<C>(coord));
  }
  else if constexpr (sized_and_shaped_and_congruent<T&&,C&&>)
  {
    return in_domain(std::forward<T>(obj), std::forward<C>(coord));
  }
  else
  {
    // obj has no shape, so there is no known domain
    // all elements are assumed to exist in this case
    return true;
  }
}


} // end ubu::detail

#include "../../detail/epilogue.hpp"

