#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinates/concepts/coordinate.hpp"
#include "../coordinates/element.hpp"
#include "terminal_element_exists.hpp"
#include <concepts>
#include <utility>

namespace ubu::detail
{


template<class T, class C>
constexpr auto recursive_element_exists(T&& obj, C&& coord);

template<class T, class C>
concept has_recursive_element_exists = requires(T obj, C coord)
{
  { detail::recursive_element_exists(std::forward<T>(obj), std::forward<C>(coord)) } -> std::convertible_to<bool>;
};


template<class T, class C>
constexpr auto recursive_element_exists(T&& obj, C&& coord)
{
  if constexpr (has_terminal_element_exists<T&&,C&&>)
  {
    // terminal case: a customization or default of element_exists(obj,coord) exists
    return terminal_element_exists(std::forward<T>(obj), std::forward<C>(coord));
  }
  else if constexpr (tuples::tuple_like_of_size_at_least<C&&,2>)
  {
    // recursive case: attempt to split the coordinate and recurse

    // split the coordinate into [leading..., last]
    auto [leading, last] = leading_and_last(std::forward<C>(coord));

    using last_t = decltype(last);
    using leading_t = decltype(leading);

    // we only enter this branch if:
    // * recursive_element_exists(obj, last) is well-formed
    // * element(obj,last) is well-formed
    // * recursive_element_exists(obj[last], leading) is well-formed
    if constexpr (has_recursive_element_exists<T&&,last_t> and
                  coordinate_for<last_t,T&&> and
                  has_recursive_element_exists<element_t<T&&,last_t>,leading_t>)
    {
      // check if the element obj[last] actually exists; if not, return false
      if(not recursive_element_exists(std::forward<T>(obj), last)) return false;

      // lookup the element obj[last]
      decltype(auto) e = element(std::forward<T>(obj), last);

      // recurse into e
      return recursive_element_exists(std::forward<decltype(e)>(e), leading);
    }
    else
    {
      // failure case: can't recurse; return void
      return;
    }
  }
  else
  {
    // failure case: C is either not a tuple or is not a large enough tuple; return void
    return;
  }
}


} // end ubu::detail

#include "../../detail/epilogue.hpp"
