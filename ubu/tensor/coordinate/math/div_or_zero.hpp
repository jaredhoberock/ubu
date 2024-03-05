#pragma once

#include "../../../detail/prologue.hpp"

#include "../concepts/integral_like.hpp"

namespace ubu
{

template<integral_like T1, integral_like T2>
constexpr integral_like auto div_or_zero(T1 dividend, T2 divisor)
{
  return divisor != 0 ? (dividend / divisor) : 0;
}

} // end ubu

#include "../../../detail/epilogue.hpp"

