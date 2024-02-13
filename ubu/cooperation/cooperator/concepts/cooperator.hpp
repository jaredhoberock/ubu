#pragma once

#include "../../../detail/prologue.hpp"
#include "../../../tensor/coordinate/coord.hpp"
#include "../../../tensor/shape/shape.hpp"
#include "../synchronize.hpp"

namespace ubu
{

template<class T>
concept cooperator = requires(T arg)
{
  coord(arg);
  shape(arg);
  synchronize(arg);
};

} // end ubu

#include "../../../detail/epilogue.hpp"

