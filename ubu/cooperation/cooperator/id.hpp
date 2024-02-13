#pragma once

#include "../../detail/prologue.hpp"

#include "../../tensor/coordinate/coord.hpp"
#include "../../tensor/layout/column_major.hpp"
#include "../../tensor/shape/shape.hpp"
#include "concepts/cooperator.hpp"
#include <concepts>

namespace ubu
{

template<cooperator C>
constexpr std::integral auto id(const C& self)
{
  column_major layout(shape(self));
  return layout[coord(self)];
}

} // end ubu

#include "../../detail/epilogue.hpp"

