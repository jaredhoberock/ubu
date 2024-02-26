#pragma once

#include "../../detail/prologue.hpp"
#include "vector_like.hpp"
#include <ranges>

namespace ubu
{

template<class V>
concept sized_vector_like =
  vector_like<V>
  and requires(V v)
  {
    std::ranges::size(v);
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"

