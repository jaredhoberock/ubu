#pragma once

#include "../../detail/prologue.hpp"
#include "../constant_valued.hpp"
#include "integral_like.hpp"

namespace ubu
{

// larger exists as an alternative to std::max without its failings
// 1. it returns a copy of its parameter, not a reference
// 2. the types of its parameters can differ
// 3. it can be passed to algorithms
constexpr inline auto larger = [](integral_like auto a, integral_like auto b)
{
  if constexpr (constant_valued<decltype(a)> and constant_valued<decltype(b)>)
  {
    if constexpr (a < b)
    {
      return b;
    }
    else
    {
      return a;
    }
  }
  else
  {
    // the if constexprs above avoid the possible conversion
    // incurred by this use of the ternary operator
    return a < b ? b : a;
  }
};

} // end ubu

#include "../../detail/epilogue.hpp"


