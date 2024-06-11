#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../miscellaneous/integral/ceil_div.hpp"
#include "../../../coordinates/concepts/coordinate.hpp"
#include "../../../coordinates/detail/tuple_algorithm.hpp"
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
  return (... and std::constructible_from<T,std::tuple_element_t<I,Tuple>>);
}


template<class T, class Tuple>
concept constructible_from_all_elements_of = constructible_from_all_elements_of_impl<T,Tuple>(tuple_indices<Tuple>);


template<tuple_like T, std::size_t... I>
  requires constructible_from_all_elements_of<std::tuple_element_t<0,T>, T>
constexpr std::array<std::tuple_element_t<0,T>, std::tuple_size_v<T>> tuple_as_array_impl(std::index_sequence<I...>, const T& t)
{
  return {{get<I>(t)...}};
}


template<tuple_like T>
  requires constructible_from_all_elements_of<std::tuple_element_t<0,T>, T>
constexpr std::array<std::tuple_element_t<0,T>, std::tuple_size_v<T>> tuple_as_array(const T& t)
{
  return tuple_as_array_impl(tuple_indices<T>, t);
}


template<class T>
constexpr auto unwrap_single(const T& arg)
{
  if constexpr (tuple_like<T>)
  {
    if constexpr (std::tuple_size_v<T> == 1)
    {
      return get<0>(arg);
    }
    else
    {
      return arg;
    }
  }
  else
  {
    return arg;
  }
}


// returns the pair (complement_shape, complement_stride)
template<coordinate S, stride_for<S> D, class CoSizeHi>
constexpr auto strided_layout_complement_impl(const S& shape, const D& stride, CoSizeHi cosize_hi)
{
  using namespace std;

  // strides_and_shapes : [(d0,s0), (d1,s1), (d2,s2), ...]
  auto strides_and_shapes = tuple_as_array(tuple_zip(as_flat_tuple(stride), as_flat_tuple(shape)));

  std::sort(strides_and_shapes.begin(), strides_and_shapes.end());

  auto [current_idx, partial_result] = tuple_fold(pair(1, pair(tuple(), tuple())), strides_and_shapes, [](auto state, auto stride_and_shape)
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

    return pair(current_idx, pair(tuple_append(partial_shape, result_shape), tuple_append(partial_stride, result_stride)));
  });

  auto [partial_shape, partial_stride] = partial_result;

  auto result_shape = tuple_append(partial_shape, ceil_div(cosize_hi, current_idx));
  auto result_stride = tuple_append(partial_stride, current_idx);

  return pair(unwrap_single(result_shape), unwrap_single(result_stride));
}


} // end ubu::detail


#include "../../../../detail/epilogue.hpp"

