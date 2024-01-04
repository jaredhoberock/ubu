#pragma once

#include "../../../detail/prologue.hpp"

#include "../rank.hpp"
#include "coordinate.hpp"

namespace ubu
{


template<class T1, class T2>
concept same_rank =
  coordinate<T1>
  and coordinate<T2>
  and (rank_v<T1> == rank_v<T2>)
;


} // end ubu

#include "../../../detail/epilogue.hpp"

