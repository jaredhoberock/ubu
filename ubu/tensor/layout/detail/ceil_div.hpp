#pragma once

#include "../../../detail/prologue.hpp"
#include <concepts>


namespace ubu::detail
{


template<std::integral T>
constexpr T ceil_div(const T& dividend, const T& divisor)
{
  return (dividend + divisor - 1) / divisor;
}


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

