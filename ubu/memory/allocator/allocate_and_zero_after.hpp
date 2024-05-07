#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "../../execution/executor/concepts/executor.hpp"
#include "../../execution/executor/execute_after.hpp"
#include "../../tensor/coordinate/concepts/coordinate.hpp"
#include "../../tensor/coordinate/detail/tuple_algorithm.hpp"
#include "../pointer/pointer_like.hpp"
#include "allocate_after.hpp"
#include "concepts/asynchronous_allocation.hpp"
#include "concepts/asynchronous_allocator.hpp"
#include "traits/allocator_value_t.hpp"
#include <cstddef>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ubu
{
namespace detail
{


// the dispatch procedure of allocate_and_zero_after is complicated
// because the ways to customization this function are subtly different
//
// dispatch looks for a customization of allocate_and_zero_after,
// which are, in decreasing priority:
//
// these are customizations that use the executor parameter
// 0. alloc.template allocate_and_zero_after<T>(exec, before, n)
// 1. allocate_and_zero_after<T>(exec, before, n)
// 2. alloc.allocate_and_zero_after(exec, before, n)
// 3. allocate_and_zero_after(alloc, exec, before, n)
//
// these are customizations that ignore the executor parameter 
// 4. alloc.template allocate_and_zero_after<T>(before, n)
// 5. allocate_and_zero_after<T>(before, n)
// 6. alloc.allocate_and_zero_after(before, n)
// 7. allocate_and_zero_after(alloc, before, n)
//
// if dispatch fails to find a customization, it uses the default:
// 8. ubu::allocate_after<T>(alloc, ...) then ubu::execute_after(exec, ...)
//
// the concepts which detect customizations 0 through 7 are below:


template<class T, class A, class E, class B, class S>
concept has_customization_0 = requires(A alloc, E exec, B before, S shape)
{
  { alloc.template allocate_and_zero_after<T>(exec, before, shape) } -> asynchronous_allocation_congruent_with<S>;
};

template<class T, class A, class E, class B , class S>
concept has_customization_1 = requires(A alloc, E exec, B before, S shape)
{
  { allocate_and_zero_after<T>(alloc, exec, before, shape) } -> asynchronous_allocation_congruent_with<S>;
};

template<class T, class A, class E, class B, class S>
concept has_customization_2 = requires(A alloc, E exec, B before, S shape)
{
  requires std::same_as<T, allocator_value_t<A>>;

  { alloc.allocate_and_zero_after(exec, before, shape) } -> asynchronous_allocation_congruent_with<S>;
};

template<class T, class A, class E, class B, class S>
concept has_customization_3 = requires(A alloc, E exec, B before, S shape)
{
  requires std::same_as<T, allocator_value_t<A>>;

  { allocate_and_zero_after(alloc, exec, before, shape) } -> asynchronous_allocation_congruent_with<S>;
};

template<class T, class A, class E, class B, class S>
concept has_customization_4 = requires(A alloc, B before, S shape)
{
  { alloc.template allocate_and_zero_after<T>(before, shape) } -> asynchronous_allocation_congruent_with<S>;
};

template<class T, class A, class E, class B , class S>
concept has_customization_5 = requires(A alloc, B before, S shape)
{
  { allocate_and_zero_after<T>(alloc, before, shape) } -> asynchronous_allocation_congruent_with<S>;
};

template<class T, class A, class E, class B, class S>
concept has_customization_6 = requires(A alloc, B before, S shape)
{
  requires std::same_as<T, allocator_value_t<A>>;

  { alloc.allocate_and_zero_after(before, shape) } -> asynchronous_allocation_congruent_with<S>;
};

template<class T, class A, class E, class B, class S>
concept has_customization_7 = requires(A alloc, B before, S shape)
{
  requires std::same_as<T, allocator_value_t<A>>;

  { allocate_and_zero_after(alloc, before, shape) } -> asynchronous_allocation_congruent_with<S>;
};


// this is the type of allocate_and_zero_after
template<class T>
struct dispatch_allocate_and_zero_after
{
  // dispatch path 0 calls a member function template with the executor
  template<class A, class E, class B, class S>
    requires has_customization_0<T, A&&, E&&, B&&, S&&>
  constexpr asynchronous_allocation auto operator()(A&& alloc, E&& exec, B&& before, S&& shape) const
  {
    return std::forward<A>(alloc).template allocate_and_zero_after<T>(std::forward<E>(exec), std::forward<B>(before), std::forward<S>(shape));
  }

  // dispatch path 1 calls a free function template with the executor
  template<class A, class E, class B, class S>
    requires (not has_customization_0<T, A&&, E&&, B&&, S&&>
              and has_customization_1<T, A&&, E&&, B&&, S&&>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, E&& exec, B&& before, S&& shape) const
  {
    return allocate_and_zero_after<T>(std::forward<A>(alloc), std::forward<E>(exec), std::forward<B>(before), std::forward<S>(shape));
  }

  // dispatch path 2 calls the member function with the executor
  template<class A, class E, class B, class S>
    requires (not has_customization_0<T, A&&, E&&, B&&, S&&>
              and not has_customization_1<T, A&&, E&&, B&&, S&&>
              and has_customization_2<T, A&&, E&&, B&&, S&&>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, E&& exec, B&& before, S&& shape) const
  {
    return std::forward<A>(alloc).allocate_and_zero_after(std::forward<E>(exec), std::forward<B>(before), std::forward<S>(shape));
  }

  // dispatch path 3 calls the free function with the executor
  template<class A, class E, class B, class S>
    requires (not has_customization_0<T, A&&, E&&, B&&, S&&>
              and not has_customization_1<T, A&&, E&&, B&&, S&&>
              and not has_customization_2<T, A&&, E&&, B&&, S&&>
              and has_customization_3<T, A&&, E&&, B&&, S&&>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, E&& exec, B&& before, S&& shape) const
  {
    return allocate_and_zero_after(std::forward<A>(alloc), std::forward<E>(exec), std::forward<B>(before), std::forward<S>(shape));
  }

  // dispatch path 4 calls a member function template without the executor
  template<class A, class E, class B, class S>
    requires (not has_customization_0<T, A&&, E&&, B&&, S&&>
              and not has_customization_1<T, A&&, E&&, B&&, S&&>
              and not has_customization_2<T, A&&, E&&, B&&, S&&>
              and not has_customization_3<T, A&&, E&&, B&&, S&&>
              and has_customization_4<T, A&&, E&&, B&&, S&&>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, E&&, B&& before, S&& shape) const
  {
    return std::forward<A>(alloc).template allocate_and_zero_after<T>(std::forward<B>(before), std::forward<S>(shape));
  }

  // dispatch path 5 calls a free function template without the executor
  template<class A, class E, class B, class S>
    requires (not has_customization_0<T, A&&, E&&, B&&, S&&>
              and not has_customization_1<T, A&&, E&&, B&&, S&&>
              and not has_customization_2<T, A&&, E&&, B&&, S&&>
              and not has_customization_3<T, A&&, E&&, B&&, S&&>
              and not has_customization_4<T, A&&, E&&, B&&, S&&>
              and has_customization_5<T, A&&, E&&, B&&, S&&>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, E&&, B&& before, S&& shape) const
  {
    return allocate_and_zero_after<T>(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape));
  }

  // dispatch path 6 calls the member function without the executor
  template<class A, class E, class B, class S>
    requires (not has_customization_0<T, A&&, E&&, B&&, S&&>
              and not has_customization_1<T, A&&, E&&, B&&, S&&>
              and not has_customization_2<T, A&&, E&&, B&&, S&&>
              and not has_customization_3<T, A&&, E&&, B&&, S&&>
              and not has_customization_4<T, A&&, E&&, B&&, S&&>
              and not has_customization_5<T, A&&, E&&, B&&, S&&>
              and has_customization_6<T, A&&, E&&, B&&, S&&>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, E&&, B&& before, S&& shape) const
  {
    return std::forward<A>(alloc).allocate_and_zero_after(std::forward<B>(before), std::forward<S>(shape));
  }

  // dispatch path 7 calls the free function without the executor
  template<class A, class E, class B, class S>
    requires (not has_customization_0<T, A&&, E&&, B&&, S&&>
              and not has_customization_1<T, A&&, E&&, B&&, S&&>
              and not has_customization_2<T, A&&, E&&, B&&, S&&>
              and not has_customization_3<T, A&&, E&&, B&&, S&&>
              and not has_customization_4<T, A&&, E&&, B&&, S&&>
              and not has_customization_5<T, A&&, E&&, B&&, S&&>
              and not has_customization_6<T, A&&, E&&, B&&, S&&>
              and has_customization_7<T, A&&, E&&, B&&, S&&>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, E&&, B&& before, S&& shape) const
  {
    return allocate_and_zero_after(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape));
  }

  // finally, dispatch path 8 is the default and calls allocate_after followed by execute_after
  template<asynchronous_allocator A, executor E, happening B, coordinate S>
    requires (not has_customization_0<T, A&&, E&&, B&&, S>
              and not has_customization_1<T, A&&, E&&, B&&, S>
              and not has_customization_2<T, A&&, E&&, B&&, S>
              and not has_customization_3<T, A&&, E&&, B&&, S>
              and not has_customization_4<T, A&&, E&&, B&&, S>
              and not has_customization_5<T, A&&, E&&, B&&, S>
              and not has_customization_6<T, A&&, E&&, B&&, S>
              and not has_customization_7<T, A&&, E&&, B&&, S>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, E&& exec, B&& before, S shape) const
  {
    // asynchronously allocate the memory
    auto [allocation_finished, span] = allocate_after<T>(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape));

    // asynchronously zero the bits
    // XXX this needs to be bulk_execute_after when we support tensor allocation
    happening auto zero_finished = execute_after(std::forward<E>(exec), std::move(allocation_finished), [ptr = span.data(), sz = span.size_bytes()]
    {
      memset(ptr, 0, sz);
    });

    // return the pair
    return std::pair(std::move(zero_finished), span);
  }
};


} // end detail


namespace
{

template<class T>
constexpr detail::dispatch_allocate_and_zero_after<T> allocate_and_zero_after;

} // end anonymous namespace


} // end ubu


#include "../../detail/epilogue.hpp"

