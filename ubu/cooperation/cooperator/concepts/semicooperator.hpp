#pragma once

#include "../../../detail/prologue.hpp"
#include "../../../tensor/coordinate/coord.hpp"
#include "../../../tensor/shape/shape.hpp"

namespace ubu
{

template<class T>
concept semicooperator = requires(T arg)
{
  coord(arg);
  shape(arg);
};

} // end ubu

#include "../../../detail/epilogue.hpp"

