#pragma once

#include "../../../detail/prologue.hpp"
#include "../../coordinate/concepts/integral_like.hpp"
#include "../../coordinate/constant.hpp"
#include <concepts>


namespace ubu::detail
{


template<integral_like T1, integral_like T2>
constexpr integral_like auto ceil_div(const T1& dividend, const T2& divisor)
{
  return (dividend + divisor - 1_c) / divisor;
}


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

