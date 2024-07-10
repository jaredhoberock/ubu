#pragma once

#include "../../detail/prologue.hpp"

#include "../../places/memory/data.hpp"
#include "sized_vector_like.hpp"

namespace ubu
{

template<class T>
concept contiguous_vector_like =
  sized_vector_like<T>
  and requires(T vec)
  {
    data(vec);
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"


