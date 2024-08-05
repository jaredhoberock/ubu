#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/traits/tensor_element.hpp"
#include "../../../causality/happening.hpp"
#include "../rebind_allocator.hpp"
#include <utility>

// the purpose of custom_deallocate_after is to call the first one of
//
// 1. alloc.deallocate_after(allocation), or
// 2. deallocate_after(alloc, allocation)
//
// which is well-formed.
//
// If neither of these is well-formed, custom_deallocate_after attempts to rebind the allocator's value_type and retries the above choices.
//
// in other words, custom_deallocate_after calls the allocator's customization of deallocate_after, if one exists, and it is allowed to
// rebind the allocator in order to locate it.

namespace ubu::detail
{

template<class A, class B, class T>
concept has_deallocate_after_member_function = requires(A alloc, B before, T tensor)
{
  { alloc.deallocate_after(before, tensor) } -> happening;
};

template<class A, class B, class T>
concept has_deallocate_after_free_function = requires(A alloc, B before, T tensor)
{
  { deallocate_after(alloc, before, tensor) } -> happening;
};

template<class A, class B, class T>
concept has_deallocate_after_customization = 
  has_deallocate_after_member_function<A,B,T>
  or has_deallocate_after_free_function<A,B,T>
;

template<class A, class B, class T>
concept has_deallocate_after_customization_once_rebound =
  has_deallocate_after_customization<A,B,T>
  or (tensor_like<T> 
      and has_rebind_allocator<tensor_element_t<T>,A>
      and has_deallocate_after_customization<
        rebind_allocator_result_t<tensor_element_t<T>,A>, B, T
      >)
;


template<class A, class B, class T>
concept has_custom_deallocate_after =
  has_deallocate_after_customization<A,B,T>
  or has_deallocate_after_customization_once_rebound<A,B,T>
;


template<class A, class B, class T>
  requires has_custom_deallocate_after<A&&,B&&,T&&>
constexpr happening auto custom_deallocate_after(A&& alloc, B&& before, T&& tensor)
{
  if constexpr (has_deallocate_after_member_function<A&&,B&&,T&&>)
  {
    return std::forward<A>(alloc).deallocate_after(std::forward<B>(before), std::forward<T>(tensor));
  }
  else if constexpr (has_deallocate_after_free_function<A&&,B&&,T&&>)
  {
    return deallocate_after(std::forward<A>(alloc), std::forward<B>(before), std::forward<T>(tensor));
  }
  else if constexpr (has_deallocate_after_customization_once_rebound<A&&,B&&,T&&>)
  {
    // rebind allocator and recurse
    using value_type = tensor_element_t<T>;

    return custom_deallocate_after(rebind_allocator<value_type>(std::forward<A>(alloc)), std::forward<B>(before), std::forward<T>(tensor));
  }
}

} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

