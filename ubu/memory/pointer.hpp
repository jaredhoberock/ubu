#pragma once

#include "../detail/prologue.hpp"

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


// XXX reference_like's requirements should be that the addressof operator
//     returns a pointer_like
//     and R should be convertible to T, which is pointer_like's pointee type
template<class R>
concept reference_like = true;


template<class P>
using pointer_pointee_t = typename std::pointer_traits<P>::element_type;


} // end ubu

#include "../detail/epilogue.hpp"

