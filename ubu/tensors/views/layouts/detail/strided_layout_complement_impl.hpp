#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../miscellaneous/integrals/ceil_div.hpp"
#include "../../../../miscellaneous/tuples.hpp"
#include "../../../coordinates/concepts/coordinate.hpp"
#include "../strided_layout.hpp"
#include "../strides/stride_for.hpp"
#include <algorithm>
#include <concepts>
#include <tuple>
#include <utility>


namespace ubu::detail
{


template<class T, class Tuple, std::size_t... I>
constexpr bool constructible_from_all_elements_of_impl(std::index_sequence<I...>)
{
  return (... and std::constructible_from<T,tuples::element_t<I,Tuple>>);
}


template<class T, class Tuple>
concept constructible_from_all_elements_of = constructible_from_all_elements_of_impl<T,Tuple>(tuples::indices_v<Tuple>);


template<tuples::tuple_like T, std::size_t... I>
  requires constructible_from_all_elements_of<tuples::element_t<0,T>, T>
constexpr std::array<tuples::element_t<0,T>, tuples::size_v<T>> tuple_as_array_impl(std::index_sequence<I...>, const T& t)
{
  return {{get<I>(t)...}};
}


template<tuples::tuple_like T>
  requires constructible_from_all_elements_of<tuples::element_t<0,T>, T>
constexpr std::array<tuples::element_t<0,T>, tuples::size_v<T>> tuple_as_array(const T& t)
{
  return tuple_as_array_impl(tuples::indices_v<T>, t);
}


// returns the pair (complement_shape, complement_stride)
template<coordinate S, stride_for<S> D, class CoSizeHi>
constexpr auto strided_layout_complement_impl(const S& shape, const D& stride, CoSizeHi cosize_hi)
{
  using namespace std;

  // strides_and_shapes : [(d0,s0), (d1,s1), (d2,s2), ...]
  auto strides_and_shapes = tuple_as_array(tuples::zip(tuples::flatten(stride), tuples::flatten(shape)));

  std::sort(strides_and_shapes.begin(), strides_and_shapes.end());

  auto [current_idx, partial_result] = tuples::fold_left(pair(1, pair(tuple(), tuple())), strides_and_shapes, [](auto state, auto stride_and_shape)
  {
    int current_idx = state.first;
    auto partial_result = state.second;

    int stride = get<0>(stride_and_shape);
    int shape = get<1>(stride_and_shape);

    int result_shape = stride / current_idx;
    int result_stride = current_idx;
    current_idx = shape * stride;

    auto partial_shape = partial_result.first;
    auto partial_stride = partial_result.second;

    return pair(current_idx, pair(tuples::append(partial_shape, result_shape), tuples::append(partial_stride, result_stride)));
  });

  auto [partial_shape, partial_stride] = partial_result;

  auto result_shape = tuples::append(partial_shape, ceil_div(cosize_hi, current_idx));
  auto result_stride = tuples::append(partial_stride, current_idx);

  return pair(tuples::unwrap_single(result_shape), tuples::unwrap_single(result_stride));
}


} // end ubu::detail


#include "../../../../detail/epilogue.hpp"

