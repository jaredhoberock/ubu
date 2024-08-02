#pragma once

#include "../../../detail/prologue.hpp"

#include "../pointers.hpp"
#include "concepts/allocator.hpp"
#include "deallocate.hpp"
#include "destroy.hpp"
#include "traits/allocator_pointer.hpp"
#include "traits/allocator_size.hpp"


namespace ubu
{

template<pointer_like P, allocator_of<pointer_pointee_t<P>> A>
constexpr void allocator_delete(A& a, P ptr)
{
  destroy(a, ptr);
  deallocate(a, ptr, 1);
}

} // end ubu

#include "../../../detail/epilogue.hpp"

