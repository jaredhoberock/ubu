#pragma once

#include "../../detail/prologue.hpp"
#include "../vector/span_like.hpp"
#include "column_major_matrix_layout.hpp"
#include "matrix_like.hpp"

namespace ubu
{

template<class T>
concept contiguous_column_major_matrix_like =
  matrix_like<T>
  and requires(T m)
  {
    // these requirements essentially identify a specific kind of ubu::view
    { m.tensor() } -> span_like;
    { m.layout() } -> column_major_matrix_layout;
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"

