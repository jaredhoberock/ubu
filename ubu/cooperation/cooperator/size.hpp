#pragma once

#include "../../detail/prologue.hpp"

#include "../../tensor/shape/shape.hpp"
#include "../../tensor/shape/shape_size.hpp"
#include "concepts/cooperator.hpp"
#include <concepts>

namespace ubu
{

template<cooperator C>
constexpr std::integral auto size(const C& self)
{
  return shape_size(shape(self));
}

} // end ubu

#include "../../detail/epilogue.hpp"

