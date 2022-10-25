#pragma once

#include "../../detail/prologue.hpp"

#include "../grid_coordinate.hpp"
#include "../element.hpp"
#include "../rank.hpp"
#include "make_coordinate.hpp"
#include "subgrid_size.hpp"
#include <concepts>
#include <utility>


namespace ubu::detail
{


template<std::integral T>
constexpr T compact_column_major_stride_impl(const std::integral auto& shape, const T& current_stride)
{
  return current_stride;
}


template<grid_coordinate S>
  requires (!std::integral<S>)
constexpr S compact_column_major_stride_impl(const S& shape, const std::integral auto& current_stride);


template<grid_coordinate S, std::size_t... Is>
constexpr S compact_column_major_stride_impl(const S& shape, const std::integral auto& current_stride, std::index_sequence<Is...>)
{
  return detail::make_coordinate<S>(detail::compact_column_major_stride_impl(element<Is>(shape), current_stride * detail::subgrid_size(shape, std::make_index_sequence<Is>{}))...);
}


template<grid_coordinate S>
  requires (!std::integral<S>)
constexpr S compact_column_major_stride_impl(const S& shape, const std::integral auto& current_stride)
{
  return detail::compact_column_major_stride_impl(shape, current_stride, std::make_index_sequence<rank_v<S>>{});
}


template<grid_coordinate S>
constexpr S compact_column_major_stride(const S& shape)
{
  return detail::compact_column_major_stride_impl(shape, 1);
}
  

} // end ubu::detail


#include "../../detail/epilogue.hpp"

