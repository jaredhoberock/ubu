#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/tuples.hpp"
#include "custom_element.hpp"
#include <type_traits>
#include <utility>

namespace ubu::detail
{


template<tuples::tuple_like_of_size_at_least<2> T>
constexpr tuples::pair_like auto leading_and_last(const T& t)
{
  // unwrap any leading singles
  return std::pair(tuples::unwrap_single(tuples::leading(t)), tuples::last(t));
}

template<tuples::tuple_like_of_size_at_least<2> T>
using leading_and_last_result_t = decltype(std::declval<T>());


template<class T, class C>
constexpr decltype(auto) recursive_element(T&& obj, C&& coord);

template<class T, class C>
concept has_recursive_element = requires(T obj, C coord)
{
  { detail::recursive_element(std::forward<T>(obj), std::forward<C>(coord)) } -> not_void;
};


// The CPO element(obj, coord) is recursively defined.
//
// the idea is that we first check for the first of the following:
//
//   1. obj.element(coord), or
//   2. element(obj,coord), or
//   3. obj[coord], or
//   4. obj(coord)
//
// If none of those work, we try again recursively if coord is a tuple with at least two elements
// If so, we split coord into (leading..., last) and recurse, looking for
//
//   1. element(obj.element(last), leading)
//   2. etc.
//   ...
//
// The purpose of this is to allow e.g. std::vector<std::vector<int>> to be indexed like this:
//
//     std::vector<std::vector<int>> nested_vec = ...
//
//     std::pair coord(inner_coord, outer_coord);
//
//     int x = element(nested_vec, coord);
//
template<class T, class C>
constexpr decltype(auto) recursive_element(T&& obj, C&& coord)
{
  if constexpr (has_custom_element<T&&,C&&>)
  {
    // terminal case: a customization of element(obj,coord) exists
    return custom_element(std::forward<T>(obj), std::forward<C>(coord));
  }
  else if constexpr (tuples::tuple_like_of_size_at_least<C&&,2>)
  {
    // recursive case: attempt to split the coordinate and recurse twice

    // split the coordinate into [leading..., last]
    auto [leading, last] = leading_and_last(std::forward<C>(coord));

    if constexpr (has_recursive_element<T&&,decltype(last)>)
    {
      // for the first lookup, recurse into the last mode of coord
      decltype(auto) first_lookup = recursive_element(std::forward<T>(obj), last);

      if constexpr (has_recursive_element<decltype(first_lookup),decltype(leading)>)
      {
        // for the second lookup, recurse into the leading modes of coord
        return recursive_element(std::forward<decltype(first_lookup)>(first_lookup), leading);
      }
      else
      {
        // error case: can't recurse into the leading modes of coord; return void
        return;
      }
    }
    else
    {
      // error case: can't recurse into the last mode of coord; return void
      return;
    }
  }
  else
  {
    // error case: C is either not a tuple or is not a large enough tuple; return void
    return;
  }
}


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

