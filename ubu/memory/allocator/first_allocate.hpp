#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/initial_happening.hpp"
#include "../../tensor/fancy_span.hpp"
#include "allocate_after.hpp"
#include "concepts/asynchronous_allocation.hpp"
#include "concepts/asynchronous_allocator.hpp"
#include "traits/allocator_value_t.hpp"
#include <memory>
#include <type_traits>
#include <utility>

namespace ubu
{
namespace detail
{


template<class T, class A, class S>
concept has_first_allocate_member_function_template = requires(A alloc, S shape)
{
  { alloc.template first_allocate<T>(shape) } -> asynchronous_allocation_congruent_with<S>;
};


template<class T, class A, class S>
concept has_first_allocate_free_function_template = requires(A alloc, S shape)
{
  { first_allocate<T>(alloc, shape) } -> asynchronous_allocation_congruent_with<S>;
};


template<class T, class A, class S>
concept has_first_allocate_member_function = requires(A alloc, S shape)
{
  requires std::same_as<T, allocator_value_t<A>>;

  { alloc.first_allocate(shape) } -> asynchronous_allocation_congruent_with<S>;
};


template<class T, class A, class S>
concept has_first_allocate_free_function = requires(A alloc, S shape)
{
  requires std::same_as<T, allocator_value_t<A>>;

  { first_allocate(alloc, shape) } -> asynchronous_allocation_congruent_with<S>;
};


template<class T, class A, class S>
concept has_first_allocate_customization =
  has_first_allocate_member_function_template<T,A,S>
  or has_first_allocate_free_function_template<T,A,S>
  or has_first_allocate_member_function<T,A,S>
  or has_first_allocate_free_function<T,A,S>
;


// this is the type of first_allocate
template<class T>
struct dispatch_first_allocate
{
  // this path calls the member function template
  template<class A, class S>
    requires has_first_allocate_member_function_template<T, A&&, S&&>
  constexpr asynchronous_allocation auto operator()(A&& alloc, S&& shape) const
  {
    return std::forward<A>(alloc).template first_allocate<T>(std::forward<S>(shape));
  }
  
  // this path calls the free function template
  template<class A, class S>
    requires (not has_first_allocate_member_function_template<T, A&&, S&&>
              and has_first_allocate_free_function_template<T, A&&, S&&>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, S&& shape) const
  {
    return first_allocate<T>(std::forward<A>(alloc), std::forward<S>(shape));
  }

  // this path calls the member function
  template<class A, class S>
    requires (not has_first_allocate_member_function_template<T, A&&, S&&>
              and not has_first_allocate_free_function_template<T, A&&, S&&>
              and has_first_allocate_member_function<T, A&&, S&&>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, S&& shape) const
  {
    return std::forward<A>(alloc).first_allocate(std::forward<S>(shape));
  }

  // this path calls the free function
  template<class A, class S>
    requires (not has_first_allocate_member_function_template<T, A&&, S&&>
              and not has_first_allocate_free_function_template<T, A&&, S&&>
              and not has_first_allocate_member_function<T, A&&, S&&>
              and has_first_allocate_free_function<T, A&&, S&&>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, S&& shape) const
  {
    return first_allocate(std::forward<A>(alloc), std::forward<S>(shape));
  }

  // this dispatch path tries to rebind and then call first_allocate again
  template<class A, class S>
    requires (not has_first_allocate_customization<T,A&&,S&&>
              and has_first_allocate_customization<T, rebind_allocator_result_t<T,A&&>, S&&>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, S&& shape) const
  {
    auto rebound_alloc = rebind_allocator<T>(std::forward<A>(alloc));
    return (*this)(rebound_alloc, std::forward<S>(shape));
  }

  // this dispatch path calls allocate_after
  template<asynchronous_allocator A, coordinate S>
    requires (not has_first_allocate_customization<T, A&&, S>
              and not has_first_allocate_customization<T, rebind_allocator_result_t<T,A&&>, S>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, S shape) const
  {
    auto [after, ptr] = allocate_after<T>(std::forward<A>(alloc), initial_happening(alloc), shape);
    return std::pair(std::move(after), fancy_span(ptr, shape));
  }
};


} // end detail


namespace
{

template<class T>
constexpr detail::dispatch_first_allocate<T> first_allocate;

} // end anonymous namespace


template<class T, class A, class S>
using first_allocate_result_t = decltype(first_allocate<T>(std::declval<A>(), std::declval<S>()));


} // end ubu

#include "../../detail/epilogue.hpp"

