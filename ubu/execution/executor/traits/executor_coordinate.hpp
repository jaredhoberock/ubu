#pragma once

#include "../../../detail/prologue.hpp"

#include "../bulk_execution_grid.hpp"
#include "../concepts/executor.hpp"

namespace ubu
{

template<class E>
  requires executor<E>
using executor_coordinate_t = decltype(bulk_execution_grid(std::declval<E>(), std::declval<std::size_t>()));

} // end ubu

#include "../../../detail/epilogue.hpp"

