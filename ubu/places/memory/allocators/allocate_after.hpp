#pragma once

#include "../../../detail/prologue.hpp"

#include "concepts/asynchronous_allocation.hpp"
#include "rebind_allocator.hpp"
#include "traits/allocator_value_t.hpp"
#include <utility>

namespace ubu
{

namespace detail
{


template<class T, class A, class B, class S>
concept has_allocate_after_member_function_template = requires(A alloc, B before, S shape)
{
  { std::forward<A>(alloc).template allocate_after<T>(std::forward<B>(before), std::forward<S>(shape)) } -> asynchronous_allocation_congruent_with<S>;
};


template<class T, class A, class B , class S>
concept has_allocate_after_free_function_template = requires(A alloc, B before, S shape)
{
  { allocate_after<T>(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape)) } -> asynchronous_allocation_congruent_with<S>;
};


template<class T, class A, class B, class S>
concept has_allocate_after_member_function = requires(A alloc, B before, S shape)
{
  requires std::same_as<T, allocator_value_t<A>>;

  { std::forward<A>(alloc).allocate_after(std::forward<B>(before), std::forward<S>(shape)) } -> asynchronous_allocation_congruent_with<S>;
};


template<class T, class A, class B, class S>
concept has_allocate_after_free_function = requires(A alloc, B before, S shape)
{
  requires std::same_as<T, allocator_value_t<A>>;

  { allocate_after(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape)) } -> asynchronous_allocation_congruent_with<S>;
};


template<class T, class A, class B, class S>
concept has_allocate_after_customization =
  has_allocate_after_member_function_template<T,A,B,S>
  or has_allocate_after_free_function_template<T,A,B,S>
  or has_allocate_after_member_function<T,A,B,S>
  or has_allocate_after_free_function<T,A,B,S>
;


// this is the type of allocate_after
template<class T>
struct dispatch_allocate_after
{
  // this dispatch path calls a member function template
  template<class A, class B, class S>
    requires has_allocate_after_member_function_template<T, A&&, B&&, S&&>
  constexpr asynchronous_allocation auto operator()(A&& alloc, B&& before, S&& shape) const
  {
    return std::forward<A>(alloc).template allocate_after<T>(std::forward<B>(before), std::forward<S>(shape));
  }

  // this dispatch path calls a free function template
  template<class A, class B, class S>
    requires (not has_allocate_after_member_function_template<T, A&&, B&&, S&&>
              and has_allocate_after_free_function_template<T, A&&, B&&, S&&>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, B&& before, S&& shape) const
  {
    return allocate_after<T>(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape));
  }

  // this dispatch path calls the member function
  template<class A, class B, class S>
    requires (not has_allocate_after_member_function_template<T, A&&, B&&, S&&>
              and not has_allocate_after_free_function_template<T, A&&, B&&, S&&>
              and has_allocate_after_member_function<T, A&&, B&&, S&&>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, B&& before, S&& shape) const
  {
    return std::forward<A>(alloc).allocate_after(std::forward<B>(before), std::forward<S>(shape));
  }

  // this dispatch path calls the free function
  template<class A, class B, class S>
    requires (not has_allocate_after_member_function_template<T, A&&, B&&, S&&>
              and not has_allocate_after_free_function_template<T, A&&, B&&, S&&>
              and not has_allocate_after_member_function<T, A&&, B&&, S&&>
              and has_allocate_after_free_function<T, A&&, B&&, S&&>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, B&& before, S&& shape) const
  {
    return allocate_after(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape));
  }

  // this dispatch path rebinds the allocator and recurses
  template<class A, class B, class S>
    requires (not has_allocate_after_customization<T, A&&, B&&, S&&>
              and has_rebind_allocator<T, A&&>
              and has_allocate_after_customization<
                T, rebind_allocator_result_t<T,A&&>, B&&, S&&
              >)
  constexpr asynchronous_allocation auto operator()(A&& alloc, B&& before, S&& shape) const
  {
    auto rebound_alloc = rebind_allocator<T>(std::forward<A>(alloc));
    return (*this)(rebound_alloc, std::forward<B>(before), std::forward<S>(shape));
  }
};


} // end detail


template<class T>
inline constexpr detail::dispatch_allocate_after<T> allocate_after;


template<class T, class A, class B, class S>
using allocate_after_result_t = decltype(allocate_after<T>(std::declval<A>(), std::declval<B>(), std::declval<S>()));


} // end ubu


#include "../../../detail/epilogue.hpp"

