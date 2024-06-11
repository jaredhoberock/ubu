#pragma once

#include "../../detail/prologue.hpp"

#include "../../places/memory/pointer/pointer_like.hpp"
#include "sized_vector_like.hpp"
#include <concepts>
#include <iterator> // for std::data

namespace ubu
{

template<class T>
concept contiguous_vector_like =
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


