#pragma once

#include "../../../detail/prologue.hpp"

#include "congruent.hpp"
#include "subdimensional.hpp"

namespace ubu
{


template<class T1, class T2>
concept strictly_subdimensional =
  subdimensional<T1,T2>
  and not congruent<T1,T2>
;


} // end ubu

#include "../../../detail/epilogue.hpp"

