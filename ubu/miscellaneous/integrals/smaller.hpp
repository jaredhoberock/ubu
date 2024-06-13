#pragma once

#include "../../detail/prologue.hpp"
#include "../constant_valued.hpp"
#include "integral_like.hpp"

namespace ubu
{

// smaller exists as an alternative to std::min without its failings
// 1. it returns a copy of its parameter, not a reference
// 2. the types of its parameters can differ
// 3. it can be passed to algorithms
constexpr inline auto smaller = [](integral_like auto a, integral_like auto b)
{
  if constexpr (constant_valued<decltype(a)> and constant_valued<decltype(b)>)
  {
    if constexpr (b < a)
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
    return b < a ? b : a;
  }
};

} // end ubu

#include "../../detail/epilogue.hpp"

