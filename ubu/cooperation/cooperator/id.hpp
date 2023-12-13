#pragma once

#include "../../detail/prologue.hpp"

#include "../../grid/coordinate/coord.hpp"
#include "../../grid/layout/column_major.hpp"
#include "../../grid/shape/shape.hpp"
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

