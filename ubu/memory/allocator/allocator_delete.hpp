#pragma once

#include "../../detail/prologue.hpp"

#include "../pointer.hpp"
#include "allocator.hpp"
#include "deallocate.hpp"
#include "destroy.hpp"
#include "traits/allocator_pointer_t.hpp"
#include "traits/allocator_size_t.hpp"


namespace ubu
{

template<pointer_like P, allocator_of<pointer_pointee_t<P>> A>
constexpr void allocator_delete(A& a, P ptr)
{
  destroy(a, ptr);
  deallocate(a, ptr, 1);
}

} // end ubu

#include "../../detail/epilogue.hpp"

