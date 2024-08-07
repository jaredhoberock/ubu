#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/tuples.hpp"
#include "../concepts/decomposable.hpp"
#include "../vectors/span_like.hpp"
#include "column_major_like.hpp"
#include "matrix_like.hpp"

namespace ubu
{

template<class T>
concept contiguous_column_major_matrix_like =
  matrix_like<T>
  and decomposable<T>
  and span_like<tuples::first_t<decompose_result_t<T>>>
  and column_major_like<tuples::second_t<decompose_result_t<T>>>
;

} // end ubu

#include "../../detail/epilogue.hpp"

