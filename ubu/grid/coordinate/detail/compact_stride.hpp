#pragma once

#include "../../../detail/prologue.hpp"

#include "../coordinate.hpp"
#include "../element.hpp"
#include "../grid_size.hpp"
#include "../rank.hpp"
#include "../same_rank.hpp"
#include "colexicographic_coordinate.hpp"
#include "tuple_algorithm.hpp"
#include <concepts>
#include <tuple>
#include <utility>


namespace ubu::detail
{


template<scalar_coordinate D, scalar_coordinate S>
constexpr D compact_stride_impl(const D& current_stride, const S&)
{
  return element<0>(current_stride);
}

template<nonscalar_coordinate D, nonscalar_coordinate S>
  requires same_rank<D,S>
constexpr S compact_stride_impl(const D& current_stride, const S& shape)
{
  return detail::tuple_zip_with(current_stride, shape, [](const auto& cs, const auto& s)
  {
    return compact_stride_impl(cs, s);
  });
}


// when we can detect that S sorts colexicographically, we fold from the left
template<scalar_coordinate D, nonscalar_coordinate S>
  requires colexicographic_coordinate<S>
constexpr S compact_stride_impl(const D& current_stride, const S& shape)
{
  auto [_,result] = detail::tuple_fold(std::pair(current_stride, std::tuple()), shape, [](auto prev, auto s)
  {
    auto [current_stride, prev_result] = prev;

    // _similar_to<S> ensures we result in a tuple similar to S
    auto result = detail::tuple_append_similar_to<S>(prev_result, compact_stride_impl(current_stride, s));

    return std::pair{current_stride * grid_size(s), result};
  });

  return result;
}


// when we cannot detect that S sorts colexicographically, we fold from the right
template<scalar_coordinate D, nonscalar_coordinate S>
  requires (not colexicographic_coordinate<S>)
constexpr S compact_stride_impl(const D& current_stride, const S& shape)
{
  auto [_,result] = detail::tuple_fold_right(std::pair(current_stride, std::tuple()), shape, [](auto prev, auto s)
  {
    auto [current_stride, prev_result] = prev;

    // _similar_to<S> ensures we result in a tuple similar to S
    auto result = detail::tuple_prepend_similar_to<S>(prev_result, compact_stride_impl(current_stride, s));

    return std::pair{current_stride * grid_size(s), result};
  });

  return result;
}


template<coordinate S>
constexpr S compact_stride(const S& shape)
{
  return compact_stride_impl(1, shape);
}
  

} // end ubu::detail


#include "../../../detail/epilogue.hpp"

