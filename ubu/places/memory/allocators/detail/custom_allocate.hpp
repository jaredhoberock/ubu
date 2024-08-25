#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/coordinates/concepts/coordinate.hpp"
#include "../../../../tensors/vectors/fancy_span.hpp"
#include "../../pointers/pointer_like.hpp"
#include "../../views/memory_view.hpp"
#include "../rebind_allocator.hpp"
#include <utility>


namespace ubu::detail
{

// the purpose of custom_allocate is to call the first one of
//
// 1. alloc.template allocate<T>(args...), or
// 2. allocate<T>(alloc, args...), or
// 3. alloc.allocate(args...), or
// 4. allocate(alloc, args...)
//
// which is well-formed.
//
// If none of these is well-formed, custom_allocate attempts to rebind the allocator's value_type and retries the above choices.
//
// in other words, custom_allocate calls the allocator's customization of allocate, if one exists, and it is allowed to
// rebind the allocator in order to locate it.


// in order to interoperate with legacy allocators, we allow the
// allocator's customization of allocate to return a pointer rather than a view
// this case is only occurs when the the requested shape of the allocation is rank-1
template<class T, class E, class S>
concept allocation_of =
  coordinate<S> and
  (memory_view_of<T,E,S> or (rank_v<S> == 1 and pointer_like_to<T,E>))
;



template<class T, class A, class S>
concept has_allocate_member_function_template = requires(A alloc, S shape)
{
  { std::forward<A>(alloc).template allocate<T>(std::forward<S>(shape)) } -> allocation_of<T,S>;
};

template<class T, class A, class S>
concept has_allocate_free_function_template = requires(A alloc, S shape)
{
  { allocate<T>(std::forward<A>(alloc), std::forward<S>(shape)) } -> allocation_of<T,S>;
};

template<class T, class A, class S>
concept has_allocate_member_function = requires(A alloc, S shape)
{
  { std::forward<A>(alloc).allocate(std::forward<S>(shape)) } -> allocation_of<T,S>;
};

template<class T, class A, class S>
concept has_allocate_free_function = requires(A alloc, S shape)
{
  { allocate(std::forward<A>(alloc), std::forward<S>(shape)) } -> allocation_of<T,S>;
};


template<class T, class A, class S>
concept has_allocate_customization =
  has_allocate_member_function_template<T,A,S> or
  has_allocate_free_function_template<T,A,S> or
  has_allocate_member_function<T,A,S> or
  has_allocate_free_function<T,A,S>
;


template<class T, class A, class S>
concept has_allocate_customization_once_rebound =
  has_rebind_allocator<T,A> and
  has_allocate_customization<
    T, rebind_allocator_result_t<T,A>, S
  >
;


template<class T, class A, class S>
concept has_custom_allocate =
  has_allocate_customization<T,A,S> or
  has_allocate_customization_once_rebound<T,A,S>
;


template<class T, class A, class S>
  requires has_custom_allocate<T,A&&,S&&>
constexpr memory_view_of<T,S> auto custom_allocate(A&& alloc, S&& shape)
{
  // this lambda will locate alloc's customization of allocate and call it
  auto lambda = [&]
  {
    if constexpr (has_allocate_member_function_template<T,A&&,S&&>)
    {
      return std::forward<A>(alloc).template allocate<T>(std::forward<S>(shape));
    }
    else if constexpr (has_allocate_free_function_template<T,A&&,S&&>)
    {
      return allocate<T>(std::forward<A>(alloc), std::forward<S>(shape));
    }
    else if constexpr (has_allocate_member_function<T,A&&,S&&>)
    {
      return std::forward<A>(alloc).allocate(std::forward<S>(shape));
    }
    else if constexpr (has_allocate_free_function<T,A&&,S&&>)
    {
      return allocate(std::forward<A>(alloc), std::forward<S>(shape));
    }
    else if constexpr (has_allocate_customization_once_rebound<T,A&&,S&&>)
    {
      // rebind the allocator and recurse
      return custom_allocate<T>(rebind_allocator<T>(std::forward<A>(alloc)), std::forward<S>(shape));
    }
  };

  // get the result from the allocator
  auto result = lambda();

  // if the allocator returned a pointer, that means the shape is rank-1.
  if constexpr (pointer_like<decltype(result)>)
  {
    // wrap the pointer in a fancy span
    return fancy_span(result,shape);
  }
  else
  {
    // in all other cases, the allocator's customization has to return a view
    return result;
  }
}
  

} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

