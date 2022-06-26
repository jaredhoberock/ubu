#pragma once

#include "../detail/prologue.hpp"

#include "executor/executor_associate.hpp"
#include "../memory/allocator/allocator_associate.hpp"

namespace ubu
{

template<class T>
concept execution_policy =
  executor_associate<T> and
  allocator_associate<T>
;

} // end ubu

#include "../detail/epilogue.hpp"

