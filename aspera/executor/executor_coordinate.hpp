#pragma once

#include "../detail/prologue.hpp"

#include "bulk_execution_grid.hpp"
#include "executor.hpp"

ASPERA_NAMESPACE_OPEN_BRACE

template<class E>
  requires executor<E>
using executor_coordinate_t = decltype(bulk_execution_grid(std::declval<E>(), std::declval<std::size_t>()));

ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

