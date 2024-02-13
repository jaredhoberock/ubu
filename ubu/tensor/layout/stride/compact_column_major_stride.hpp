#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinate/concepts/coordinate.hpp"
#include "../../coordinate/concepts/same_rank.hpp"
#include "../../coordinate/detail/as_integral.hpp"
#include "../../coordinate/detail/tuple_algorithm.hpp"
#include "../../coordinate/rank.hpp"
#include "../../shape/shape_size.hpp"
#include <concepts>
#include <tuple>
#include <utility>


namespace ubu
{
namespace detail
{


template<scalar_coordinate D, scalar_coordinate S>
constexpr D compact_column_major_stride_impl(const D& current_stride, const S&)
{
  return as_integral(current_stride);
}

template<nonscalar_coordinate D, nonscalar_coordinate S>
  requires same_rank<D,S>
constexpr S compact_column_major_stride_impl(const D& current_stride, const S& shape)
{
  return tuple_zip_with(current_stride, shape, [](const auto& cs, const auto& s)
  {
    return compact_column_major_stride_impl(cs, s);
  });
}

template<scalar_coordinate D, nonscalar_coordinate S>
constexpr S compact_column_major_stride_impl(const D& current_stride, const S& shape)
{
  auto [_,result] = tuple_fold(std::pair(current_stride, std::tuple()), shape, [](auto prev, auto s)
  {
    auto [current_stride, prev_result] = prev;
    auto result = tuple_append_similar_to<S>(prev_result, compact_column_major_stride_impl(current_stride, s));

    return std::pair{current_stride * shape_size(s), result};
  });

  return result;
}


} // end detail


template<coordinate S>
constexpr S compact_column_major_stride(const S& shape)
{
  return detail::compact_column_major_stride_impl(1, shape);
}
  

} // end ubu::detail


#include "../../../detail/epilogue.hpp"

