#pragma once

#include "../../detail/prologue.hpp"

#include "../../miscellaneous/size.hpp"
#include "barrier_like.hpp"
#include <ranges>

namespace ubu
{

template<class T>
concept sized_barrier_like = barrier_like<T> and sized<T>;

} // end ubu

#include "../../detail/epilogue.hpp"

