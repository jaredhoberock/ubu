#pragma once

#include "../detail/prologue.hpp"
#include "../tensor/coordinate/concepts/integral_like.hpp"
#include "constant_valued.hpp"

namespace ubu
{

// smaller exists as an alternative to std::min without its failings
template<integral_like A, integral_like B>
constexpr integral_like auto smaller(A a, B b)
{
  if constexpr (constant_valued<A> and constant_valued<B>)
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
}

} // end ubu

#include "../detail/epilogue.hpp"

