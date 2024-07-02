#pragma once

#include "../detail/prologue.hpp"
#include "concepts/tensor_like.hpp"
#include "shapes/shape.hpp"
#include "shapes/shape_size.hpp"

namespace ubu
{

template<tensor_like T>
constexpr bool empty(T&& tensor)
{
  return ubu::shape_size(ubu::shape(tensor)) == 0;
}

} // end ubu

#include "../detail/epilogue.hpp"

