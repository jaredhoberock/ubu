#pragma once

#include "../detail/prologue.hpp"

#include "congruent.hpp"
#include "element.hpp"
#include "size.hpp"
#include "detail/make_coordinate.hpp"
#include <concepts>
#include <type_traits>


namespace ubu
{


template<std::integral T>
constexpr T to_grid_coordinate(const std::integral auto& i, const std::integral auto& shape, const std::integral auto& stride)
{
  return (i / stride) % shape;
}


template<grid_coordinate T, grid_coordinate Shape, grid_coordinate Stride>
  requires (!std::integral<Shape> and congruent<T,Shape,Stride>)
constexpr T to_grid_coordinate(const std::integral auto& i, const Shape& shape, const Stride& stride);


namespace detail
{


template<grid_coordinate T, grid_coordinate Shape, grid_coordinate Stride, std::size_t... Is>
  requires (!std::integral<Shape> and congruent<T,Shape,Stride>)
constexpr T to_grid_coordinate_impl(const std::integral auto& i, const Shape& shape, const Stride& stride, std::index_sequence<Is...>)
{
  return detail::make_coordinate<T>(ubu::to_grid_coordinate<element_t<Is, T>>(i, element<Is>(shape), element<Is>(stride))...);
}


} // end detail


template<grid_coordinate T, grid_coordinate Shape, grid_coordinate Stride>
  requires (!std::integral<Shape> and congruent<T,Shape,Stride>)
constexpr T to_grid_coordinate(const std::integral auto& i, const Shape& shape, const Stride& stride)
{
  return detail::to_grid_coordinate_impl<T>(i, shape, stride, std::make_index_sequence<size_v<Shape>>{});
}


} // end ubu


#include "../detail/epilogue.hpp"

