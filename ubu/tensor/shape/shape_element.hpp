#pragma once

#include "../../detail/prologue.hpp"
#include "../coordinate/traits/coordinate_element.hpp"
#include "../coordinate/traits/rank.hpp"
#include "shape.hpp"

namespace ubu
{

template<std::size_t I, shaped T>
  requires (I < rank_v<shape_t<T>>)
using shape_element_t = coordinate_element_t<I,shape_t<T>>;

template<std::size_t I, shaped T>
  requires (I < rank_v<shape_t<T>>) and constant_valued<shape_element_t<I,T>>
constexpr inline auto shape_element_v = coordinate_element_v<I,shape_t<T>>;

} // end ubu

#include "../../detail/epilogue.hpp"

