#pragma once

#include "../../detail/prologue.hpp"

#include "integral_like.hpp"
#include "../../tensor/coordinate/constant.hpp"

namespace ubu
{


template<integral_like T1, integral_like T2>
constexpr integral_like auto ceil_div(T1 dividend, T2 divisor)
{
  return (dividend + divisor - 1_c) / divisor;
}


} // end ubu

#include "../../detail/epilogue.hpp"

