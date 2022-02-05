#pragma once

#include "../detail/prologue.hpp"

#include "future.hpp"
#include <utility>

ASPERA_NAMESPACE_OPEN_BRACE

template<class F>
  requires future<F>
using future_result_t = decltype(std::declval<F>().get());

ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

