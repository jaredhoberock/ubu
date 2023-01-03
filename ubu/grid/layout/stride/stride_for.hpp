#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinate/weakly_congruent.hpp"


namespace ubu
{


template<class D, class S>
concept stride_for = weakly_congruent<S,D>;


} // end ubu

#include "../../../detail/epilogue.hpp"

