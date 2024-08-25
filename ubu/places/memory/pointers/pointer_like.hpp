#pragma once

#include "../../../detail/prologue.hpp"

#include <concepts>
#include <iterator>
#include <memory>

namespace ubu
{


// XXX pointer_like's requirements should be something like
//     this thing is a contiguous iterator and
//     its dereference operator returns a reference_like thing
//     however, we can't simply use std::contiguous_iterator because
//     it requires that dereference returns a raw reference
//     also, it would exclude void *
template<class P>
concept pointer_like = 
  requires
  {
    typename std::pointer_traits<P>::element_type;
  }

  // just random_access_iterator would exclude void *
  and (std::is_pointer_v<P> or std::random_access_iterator<P>)
;


template<class P, class E>
concept pointer_like_to = 
  pointer_like<P> and
  std::same_as<typename std::pointer_traits<P>::element_type,E>
;


} // end ubu

#include "../../../detail/epilogue.hpp"

