#pragma once

#include "../../detail/prologue.hpp"

#include "allocator.hpp"
#include "deallocate.hpp"
#include "destroy.hpp"
#include "traits/allocator_pointer_t.hpp"
#include "traits/allocator_size_t.hpp"


ASPERA_NAMESPACE_OPEN_BRACE


template<allocator A>
constexpr void allocator_delete(A& a, allocator_pointer_t<A> ptr)
{
  destroy(a, ptr);
  deallocate(a, ptr, 1);
}


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

