#pragma once

#include "../../detail/prologue.hpp"
#include "../coordinates/element.hpp"
#include "../coordinates/traits/coordinate_element.hpp"
#include "../coordinates/traits/rank.hpp"
#include "shape.hpp"

namespace ubu
{

template<std::size_t I, shaped T>
  requires (I < rank_v<shape_t<T>>)
constexpr coordinate_element_t<I,shape_t<T>> shape_element(const T& arg)
{
  return element(shape(arg), constant<I>());
}

template<std::size_t I, shaped T>
  requires (I < rank_v<shape_t<T>>)
using shape_element_t = decltype(shape_element<I>(std::declval<T>()));

template<std::size_t I, shaped T>
  requires (I < rank_v<shape_t<T>>) and constant_valued<shape_element_t<I,T>>
constexpr inline auto shape_element_v = coordinate_element_v<I,shape_t<T>>;

} // end ubu

#include "../../detail/epilogue.hpp"

