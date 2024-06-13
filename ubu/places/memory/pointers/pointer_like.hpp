#pragma once

#include "../../../detail/prologue.hpp"

#include <memory>

namespace ubu
{


// XXX pointer_like's requirements should be something like
//     this thing is a contiguous iterator and
//     its dereference operator returns a reference_like thing
template<class P>
concept pointer_like = requires
{
  typename std::pointer_traits<P>::element_type;
};


} // end ubu

#include "../../../detail/epilogue.hpp"

