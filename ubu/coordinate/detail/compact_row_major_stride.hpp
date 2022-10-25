#pragma once

#include "../../detail/prologue.hpp"

#include "../element.hpp"
#include "../grid_coordinate.hpp"
#include "../rank.hpp"
#include "make_ascending_index_range.hpp"
#include "make_coordinate.hpp"
#include "subgrid_size.hpp"
#include <concepts>
#include <utility>


namespace ubu::detail
{


template<std::integral T>
constexpr T compact_row_major_stride_impl(const std::integral auto&, const T& current_stride)
{
  return current_stride;
}


// forward declaration for recursive case
template<tuple_like_grid_coordinate T>
constexpr T compact_row_major_stride_impl(const T& shape, const std::integral auto& current_stride);


template<grid_coordinate T, std::size_t... Is>
constexpr T compact_row_major_stride_impl(const T& shape, const std::integral auto& current_stride, std::index_sequence<Is...>)
{
  return detail::make_coordinate<T>(detail::compact_row_major_stride_impl(element<Is>(shape), current_stride * detail::subgrid_size(shape, detail::make_ascending_index_range<Is+1,rank_v<T>>{}))...);
}


template<tuple_like_grid_coordinate T>
constexpr T compact_row_major_stride_impl(const T& shape, const std::integral auto& current_stride)
{
  return detail::compact_row_major_stride_impl(shape, current_stride, std::make_index_sequence<rank_v<T>>{});
}


template<grid_coordinate T>
constexpr T compact_row_major_stride(const T& shape)
{
  return detail::compact_row_major_stride_impl(shape, 1);
}
  

} // end ubu::detail


#include "../../detail/epilogue.hpp"
