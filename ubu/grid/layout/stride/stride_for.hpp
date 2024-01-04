#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinate/concepts/weakly_congruent.hpp"


namespace ubu
{


template<class S, class C>
concept stride_for = weakly_congruent<C,S>;


} // end ubu

#include "../../../detail/epilogue.hpp"

