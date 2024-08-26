#pragma once

#include "../../../../detail/prologue.hpp"
#include "../../../../tensors/traits/tensor_element.hpp"
#include "../../../../tensors/vectors/span_like.hpp"
#include "../../../../utilities/integrals/size.hpp"
#include "../../data.hpp"
#include "../rebind_allocator.hpp"
#include <type_traits>
#include <utility>


namespace ubu::detail
{

// the purpose of custom_deallocate is to call the first one of
//
// 1. alloc.deallocate(args...), or
// 2. deallocate(alloc, args...)
//
// which is well-formed.
//
// If none of these is well-formed, custom_deallocate attempts to rebind the allocator's value_type and retries the above choices.
//
// in other words, custom_deallocate calls the allocator's customization of deallocate, if one exists, and it is allowed to
// rebind the allocator in order to locate it.


template<class A, class V>
concept has_deallocate_member_function = requires(A alloc, V view)
{
  std::forward<A>(alloc).deallocate(std::forward<V>(view));
};

template<class A, class V>
concept has_deallocate_free_function = requires(A alloc, V view)
{
  deallocate(std::forward<A>(alloc), std::forward<V>(view));
};

template<class A, class P, class S>
concept has_deallocate_ptr_size_member_function = requires(A alloc, P ptr, S size)
{
  std::forward<A>(alloc).deallocate(std::forward<P>(ptr), std::forward<S>(size));
};

template<class A, class P, class S>
concept has_deallocate_ptr_size_free_function = requires(A alloc, P ptr, S size)
{
  deallocate(std::forward<A>(alloc), std::forward<P>(ptr), std::forward<S>(size));
};


template<class A, class V>
concept has_deallocate_customization =
  has_deallocate_member_function<A,V> or
  has_deallocate_free_function<A,V> or
  (
     span_like<std::remove_cvref_t<V>> and
     (
        has_deallocate_ptr_size_member_function<A,data_t<V>,size_result_t<V>> or
        has_deallocate_ptr_size_free_function<A,data_t<V>,size_result_t<V>>
     )
  )
;


template<class A, class V>
concept has_deallocate_customization_once_rebound =
  has_rebind_allocator<tensor_element_t<V>,A> and
  has_deallocate_customization<
    rebind_allocator_result_t<tensor_element_t<V>,A>, V
  >
;


template<class A, class V>
concept has_custom_deallocate =
  has_deallocate_customization<A,V> or
  has_deallocate_customization_once_rebound<A,V>
;



template<class A, class V>
  requires has_custom_deallocate<A&&,V&&>
constexpr void custom_deallocate(A&& alloc, V&& view)
{
  constexpr bool is_span = span_like<std::remove_cvref_t<V>>;
  
  if constexpr (has_deallocate_member_function<A&&,V&&>)
  {
    return std::forward<A>(alloc).deallocate(std::forward<V>(view));
  }
  else if constexpr (has_deallocate_free_function<A&&,V&&>)
  {
    return deallocate(std::forward<A>(alloc), std::forward<V>(view));
  }
  else if constexpr (is_span)
  {
    // check if alloc is a legacy allocator that takes (ptr, sz)

    auto ptr = ubu::data(std::forward<V>(view));
    auto sz = ubu::size(view);

    using P = decltype(ptr);
    using S = decltype(sz);

    if constexpr (has_deallocate_ptr_size_member_function<A&&,P,S>)
    {
      return std::forward<A>(alloc).deallocate(ptr, sz);
    }
    else if constexpr (has_deallocate_ptr_size_free_function<A&&,P,S>)
    {
      return deallocate(std::forward<A>(alloc), ptr, sz);
    }
    else
    {
      // rebind the allocator and recurse
      using T = tensor_element_t<V&&>;

      return custom_deallocate(rebind_allocator<T>(std::forward<A>(alloc)), std::forward<V>(view));
    }
  }
  else
  {
    // rebind the allocator and recurse
    using T = tensor_element_t<V&&>;

    return custom_deallocate(rebind_allocator<T>(std::forward<A>(alloc)), std::forward<V>(view));
  }
}


} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

