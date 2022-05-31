#pragma once

#include "../../../detail/prologue.hpp"

#include "../allocator.hpp"
#include <memory>

UBU_NAMESPACE_OPEN_BRACE


template<class A>
  requires allocator<A>
using allocator_value_t = typename std::allocator_traits<std::decay_t<A>>::value_type;


UBU_NAMESPACE_CLOSE_BRACE

#include "../../../detail/epilogue.hpp"

