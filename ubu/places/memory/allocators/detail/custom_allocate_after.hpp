#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../causality/happening.hpp"
#include "../concepts/asynchronous_allocation.hpp"
#include <utility>

// the purpose of custom_allocate_after is to simply call the first one of
//
// 1. alloc.template allocate_after<T>(args...), or
// 2. allocate_after<T>(alloc, args...), or
// 3. alloc.allocate_after(args...), or
// 4. allocate_after(alloc, args...)
//
// which is well-formed.
// in other words, custom_allocate_after calls the allocator's customization of allocate_after, if one exists

namespace ubu::detail
{


template<class T, class A, class B, class S>
concept has_allocate_after_member_function_template = requires(A alloc, B before, S shape)
{
  { std::forward<A>(alloc).template allocate_after<T>(std::forward<B>(before), std::forward<S>(shape)) } -> asynchronous_tensor_like<T,S>;
};


template<class T, class A, class B , class S>
concept has_allocate_after_free_function_template = requires(A alloc, B before, S shape)
{
  { allocate_after<T>(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape)) } -> asynchronous_tensor_like<T,S>;
};


template<class T, class A, class B, class S>
concept has_allocate_after_member_function = requires(A alloc, B before, S shape)
{
  { std::forward<A>(alloc).allocate_after(std::forward<B>(before), std::forward<S>(shape)) } -> asynchronous_tensor_like<T,S>;
};


template<class T, class A, class B, class S>
concept has_allocate_after_free_function = requires(A alloc, B before, S shape)
{
  { allocate_after(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape)) } -> asynchronous_tensor_like<T,S>;
};


template<class T, class A, class B, class S>
concept has_custom_allocate_after =
  has_allocate_after_member_function_template<T,A,B,S>
  or has_allocate_after_free_function_template<T,A,B,S>
  or has_allocate_after_member_function<T,A,B,S>
  or has_allocate_after_free_function<T,A,B,S>
;


template<class T, class A, class B, class S>
  requires has_custom_allocate_after<T,A&&,B&&,S&&>
constexpr asynchronous_allocation auto custom_allocate_after(A&& alloc, B&& before, S&& shape)
{
  if constexpr (has_allocate_after_member_function_template<T,A&&,B&&,S&&>)
  {
    return std::forward<A>(alloc).template allocate_after<T>(std::forward<B>(before), std::forward<S>(shape));
  }
  else if constexpr (has_allocate_after_free_function_template<T,A&&,B&&,S&&>)
  {
    return allocate_after<T>(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape));
  }
  else if constexpr (has_allocate_after_member_function<T,A&&,B&&,S&&>)
  {
    return std::forward<A>(alloc).allocate_after(std::forward<B>(before), std::forward<S>(shape));
  }
  else
  {
    return allocate_after(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape));
  }
}



} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

