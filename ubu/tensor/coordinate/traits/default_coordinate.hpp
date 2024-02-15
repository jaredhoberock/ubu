#pragma once

#include "../../../detail/prologue.hpp"
#include "../concepts/coordinate.hpp"
#include "../zeros.hpp"

namespace ubu
{

// given some shape S, what is the default type of
// coordinate we should use for a tensor with that shape?
// 
// The purpose of this trait is that some types of shape
// are constant (i.e., ubu::constant) and cannot be used
// as coordinates (because they are fixed)
//
// To deal with this, we use the type returned by zeros<S>
//
// XXX this isn't right
//     we should map as_integral across S and use that type
template<coordinate S>
using default_coordinate_t = zeros_t<S>;

} // end ubu

#include "../../../detail/epilogue.hpp"

