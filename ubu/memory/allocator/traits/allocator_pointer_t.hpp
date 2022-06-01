#pragma once

#include "../../../detail/prologue.hpp"

#include "../allocator.hpp"
#include <memory>

namespace ubu
{


template<class A>
  requires allocator<A>
using allocator_pointer_t = typename std::allocator_traits<std::decay_t<A>>::pointer;


} // end ubu

#include "../../../detail/epilogue.hpp"

