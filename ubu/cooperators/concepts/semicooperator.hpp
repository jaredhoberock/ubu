#pragma once

#include "../../detail/prologue.hpp"
#include "../../tensors/coordinates/coord.hpp"
#include "../../tensors/shapes/shape.hpp"

namespace ubu
{

template<class T>
concept semicooperator = requires(T arg)
{
  coord(arg);
  shape(arg);
};

} // end ubu

#include "../../detail/epilogue.hpp"

