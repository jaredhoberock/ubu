#pragma once

#include "../detail/prologue.hpp"

#include "congruent.hpp"
#include "element.hpp"
#include "grid_coordinate.hpp"
#include "size.hpp"
#include <concepts>
#include <cstdint>
#include <utility>

ASPERA_NAMESPACE_OPEN_BRACE


// XXX the shape parameter is never actually used in this computation... what gives?


constexpr std::size_t to_index(const std::integral auto& coord, const std::integral auto&, const std::integral auto& stride)
{
  return coord * stride;
}


// forward declaration of the recursive case of to_index
template<class T1, class T2, class T3>
  requires (!std::integral<T1> and !std::integral<T2> and !std::integral<T3> and
            are_grid_coordinates<T1,T2,T3> and
            congruent<T1,T2,T3>)
constexpr std::size_t to_index(const T1& coord, const T2& shape, const T3& stride);


namespace detail
{



template<class T1, class T2, class T3>
  requires (!std::integral<T1> and !std::integral<T2> and !std::integral<T3> and
            are_grid_coordinates<T1,T2,T3> and
            congruent<T1,T2,T3>)
constexpr std::size_t to_index_impl(const T1& coord, const T2& shape, const T3& stride, std::index_sequence<>)
{
  return 0;
}


template<class T1, class T2, class T3, std::size_t i0, std::size_t... is>
  requires (!std::integral<T1> and !std::integral<T2> and !std::integral<T3> and
            are_grid_coordinates<T1,T2,T3> and
            congruent<T1,T2,T3>)
constexpr std::size_t to_index_impl(const T1& coord, const T2& shape, const T3& stride, std::index_sequence<i0,is...>)
{
  return ASPERA_NAMESPACE::to_index(element<i0>(coord), element<i0>(shape), element<i0>(stride))
    + detail::to_index_impl(coord, shape, stride, std::index_sequence<is...>{});
}



} // end detail


template<class T1, class T2, class T3>
  requires (!std::integral<T1> and !std::integral<T2> and !std::integral<T3> and
            are_grid_coordinates<T1,T2,T3> and
            congruent<T1,T2,T3>)
constexpr std::size_t to_index(const T1& coord, const T2& shape, const T3& stride)
{
  return detail::to_index_impl(coord, shape, stride, std::make_index_sequence<size_v<T1>>{});
}


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

