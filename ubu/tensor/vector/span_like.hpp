#pragma once

#include "../../detail/prologue.hpp"
#include "../../memory/pointer/pointer_like.hpp"
#include "../iterator.hpp"
#include "sized_vector_like.hpp"
#include <concepts>
#include <span>

namespace ubu
{

// something is span-like if it is 1D, sized, and provides access to its data
// XXX we also need to require that T is a view
template<class T>
concept span_like =
  sized_vector_like<T>
  and requires(T vec)
  {
    // std::ranges::data isn't what we want because it
    // requires .data() to return a raw pointer
    { std::data(vec) } -> pointer_like;
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"

