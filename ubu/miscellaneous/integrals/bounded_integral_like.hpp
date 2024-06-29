#pragma once

#include "../../detail/prologue.hpp"

#include "integral_like.hpp"
#include "to_integral.hpp"
#include <limits>

namespace ubu
{

template<class T>
concept bounded_integral_like =
  integral_like<T>

  // we have T, the type we are testing, and I, the integral type that T converts to via to_integral
  //
  // T is bounded if the maximum value of T is smaller than the maximum value of I
  and (std::numeric_limits<T>::max() < std::numeric_limits<to_integral_t<T>>::max())
;

} // end ubu

#include "../../detail/epilogue.hpp"

