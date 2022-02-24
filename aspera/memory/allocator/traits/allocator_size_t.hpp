#pragma once

#include "../../../detail/prologue.hpp"

#include "../allocator.hpp"
#include <memory>

ASPERA_NAMESPACE_OPEN_BRACE


template<class A>
  requires allocator<A>
using allocator_size_t = typename std::allocator_traits<std::decay_t<A>>::size_type;


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../../detail/epilogue.hpp"
