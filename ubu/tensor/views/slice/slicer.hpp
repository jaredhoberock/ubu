#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../miscellaneous/integral/integral_like.hpp"
#include "../../coordinates/concepts/coordinate.hpp"
#include "../../coordinates/detail/tuple_algorithm.hpp"
#include "underscore.hpp"
#include <tuple>
#include <type_traits>

namespace ubu
{


template<class T>
concept scalar_slicer =
  detail::is_underscore_v<T>
  or scalar_coordinate<T>
;

namespace detail
{

template<class T>
struct is_nonscalar_slicer;

// check T for elements 0...N-1, and check that each one is itself a slicer
template<tuple_like T, std::size_t... I>
constexpr bool has_tuple_elements_that_are_slicers(std::index_sequence<I...>)
{
  return (... and (scalar_slicer<std::tuple_element_t<I,T>> or is_nonscalar_slicer<std::tuple_element_t<I,T>>::value));
}

template<class T>
struct is_nonscalar_slicer
{
  template<class U = T>
    requires tuple_like<U>
  static constexpr bool test(int)
  {
    return has_tuple_elements_that_are_slicers<std::remove_cvref_t<U>>(tuple_indices<U>);
  }

  static constexpr bool test(...)
  {
    return false;
  }

  static constexpr bool value = test(0);
};

} // end detail

// nonscalar_slicer is a recursive concept, so we need to implement it with traditional SFINAE techniques
template<class T>
concept nonscalar_slicer = 
  (not scalar_slicer<T>)
  and detail::is_nonscalar_slicer<T>::value
;

// a slicer is either a scalar or nonscalar slicer
template<class T>
concept slicer = scalar_slicer<T> or nonscalar_slicer<T>;

namespace detail
{

template<slicer S>
constexpr std::size_t underscore_count_v_impl()
{
  if constexpr(is_underscore_v<S>)
  {
    return 1;
  }
  else if constexpr(integral_like<S> or unit_like<S>)
  {
    return 0;
  }
  else if constexpr(tuple_like<S>)
  {
    using tuple_head_t = std::tuple_element_t<0,S>;
    using tuple_tail_t = decltype(tuple_drop_first(std::declval<S>()));

    // recurse down the head and tail and sum the results
    return underscore_count_v_impl<tuple_head_t>() + underscore_count_v_impl<tuple_tail_t>();
  }
  else
  {
    static_assert("underscore_count: bad S");
  }
}

template<slicer S>
constexpr std::size_t underscore_count_v = underscore_count_v_impl<S>();

} // end detail


// a slicer without any underscore
template<class S>
concept slicer_without_underscore =
  slicer<S>
  and (detail::underscore_count_v<S> == 0)
;

// a slicer with at least one underscore
template<class S>
concept slicer_with_underscore =
  slicer<S>
  and (detail::underscore_count_v<S> > 0)
;


namespace detail
{


template<slicer S, semicoordinate C>
constexpr bool is_slicer_for()
{
  if constexpr(scalar_slicer<S>)
  {
    // terminal case 1: S is scalar
    return true;
  }
  else if constexpr(not same_tuple_size<S,C>)
  {
    // terminal case 2: S has the wrong rank
    return false;
  }
  else
  {
    // recursive case: both S and C are tuple-like
    auto elements_are_slicers = []<std::size_t...I>(std::index_sequence<I...>)
    {
      using S_ = std::remove_cvref_t<S>;
      using C_ = std::remove_cvref_t<C>;

      return (... and is_slicer_for<std::tuple_element_t<I,S_>, std::tuple_element_t<I,C_>>());
    };

    return elements_are_slicers(tuple_indices<S>);
  }
}

} // end detail


// S is a slicer for a semicoordinate C if S is a slicer of compatible shape
template<class S, class C>
concept slicer_for =
  slicer<S>
  and semicoordinate<C>
  and detail::is_slicer_for<S,C>()
;

template<class S, class C>
concept scalar_slicer_for =
  scalar_slicer<S>
  and slicer_for<S,C>
;

template<class S, class C>
concept nonscalar_slicer_for =
  nonscalar_slicer<S>
  and slicer_for<S,C>
;


} // end ubu

#include "../../../detail/epilogue.hpp"

