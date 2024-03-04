#pragma once

#include "../../../detail/prologue.hpp"

#include "../traits/rank.hpp"
#include "semicoordinate.hpp"

namespace ubu
{


template<class T1, class T2>
concept equal_rank =
  semicoordinate<T1>
  and semicoordinate<T2>
  and (rank_v<T1> == rank_v<T2>)
;


} // end ubu

#include "../../../detail/epilogue.hpp"

