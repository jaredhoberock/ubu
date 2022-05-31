#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "../../execution/executor/dependent_on.hpp"
#include "../../execution/executor/execute_after.hpp"
#include "../../execution/executor/executor_associate.hpp"
#include "destroy.hpp"
#include "traits/allocator_pointer_t.hpp"
#include "traits/allocator_size_t.hpp"
#include "traits/allocator_value_t.hpp"
#include <utility>


UBU_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class A, class E, class P, class N>
concept has_destroy_after_member_function = requires(A alloc, E before, P ptr, N n)
{
  alloc.destroy_after(before, ptr, n);
};


template<class A, class E, class P, class N>
concept has_destroy_after_free_function = requires(A alloc, E before, P ptr, N n)
{
  destroy_after(alloc, before, ptr, n);
};


// this is the type of destroy_after
struct dispatch_destroy_after
{
  // this dispatch path calls the member function
  template<class Alloc, class Event, class P, class N>
    requires has_destroy_after_member_function<Alloc&&, Event&&, P&&, N&&>
  constexpr auto operator()(Alloc&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    return std::forward<Alloc>(alloc).destroy_after(std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Alloc, class Event, class P, class N>
    requires (!has_destroy_after_member_function<Alloc&&, Event&&, P&&, N&&> and
               has_destroy_after_free_function<Alloc&&, Event&&, P&&, N&&>)
  constexpr auto operator()(Alloc&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    return destroy_after(std::forward<Alloc>(alloc), std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // path for objects with destructors
  template<allocator A, event E>
    requires (!has_destroy_after_member_function<A&&, E&&, allocator_pointer_t<A>, allocator_size_t<A>> and
              !has_destroy_after_free_function<A&&, E&&, allocator_pointer_t<A>, allocator_size_t<A>> and
              executor_associate<A&&> and
              !std::is_trivially_destructible_v<allocator_value_t<A>>
             )
  constexpr auto operator()(A&& alloc, E&& before, allocator_pointer_t<A> ptr, allocator_size_t<A> n) const
  {
    // get an executor
    auto ex = associated_executor(std::forward<A>(alloc));

    // execute destructors
    // XXX this needs to be a bulk_execute call
    return execute_after(ex, std::forward<E>(before), [=]
    {
      for(allocator_size_t<A> i = 0; i < n; ++i)
      {
        destroy(alloc, ptr + i);
      }
    });
  }

  // path for objects without destructors
  template<allocator A, event E>
    requires (!has_destroy_after_member_function<A&&, E&&, allocator_pointer_t<A>, allocator_size_t<A>> and
              !has_destroy_after_free_function<A&&, E&&, allocator_pointer_t<A>, allocator_size_t<A>> and
              executor_associate<A&&> and
              std::is_trivially_destructible_v<allocator_value_t<A>>
             )
  constexpr auto operator()(A&& alloc, E&& before, allocator_pointer_t<A> ptr, allocator_size_t<A> n) const
  {
    // get an executor
    auto ex = associated_executor(std::forward<A>(alloc));

    return dependent_on(ex, std::forward<E>(before));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_destroy_after destroy_after;

} // end anonymous namespace


template<class A, class E, class P, class N>
using destroy_after_result_t = decltype(UBU_NAMESPACE::destroy_after(std::declval<A>(), std::declval<E>(), std::declval<P>(), std::declval<N>()));


UBU_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

