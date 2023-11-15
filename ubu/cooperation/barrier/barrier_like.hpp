#pragma once

#include "../../detail/prologue.hpp"

#include "arrive_and_wait.hpp"

namespace ubu
{

template<class T>
concept barrier_like = requires(T bar)
{
  arrive_and_wait(bar);
};

} // end ubu

#include "../../detail/epilogue.hpp"

