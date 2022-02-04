#pragma once

#include "../../../detail/prologue.hpp"

#include "../allocator.hpp"
#include <memory>

ASPERA_NAMESPACE_OPEN_BRACE


template<class A>
  requires allocator<A>
using allocator_pointer_t = typename std::allocator_traits<std::decay_t<A>>::pointer;


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../../detail/epilogue.hpp"

