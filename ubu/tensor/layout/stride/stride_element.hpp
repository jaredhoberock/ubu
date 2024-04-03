#pragma once

#include "../../../detail/prologue.hpp"
#include "../../coordinate/traits/coordinate_element.hpp"
#include "../../coordinate/traits/rank.hpp"
#include "stride.hpp"

namespace ubu
{

template<std::size_t I, strided T>
  requires (I < rank_v<stride_t<T>>)
using stride_element_t = coordinate_element_t<I,stride_t<T>>;

template<std::size_t I, strided T>
  requires (I < rank_v<stride_t<T>>) and constant_valued<stride_element_t<I,T>>
constexpr inline auto stride_element_v = coordinate_element_v<I,stride_t<T>>;

} // end ubu

#include "../../../detail/epilogue.hpp"

