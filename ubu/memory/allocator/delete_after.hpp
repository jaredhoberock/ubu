#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "../allocator/asynchronous_allocator.hpp"
#include "../allocator/deallocate_after.hpp"
#include "../allocator/destroy_after.hpp"
#include <utility>


namespace ubu
{

namespace detail
{


template<class A, class E, class P, class N>
concept has_delete_after_member_function = requires(A alloc, E before, P ptr, N n)
{
  {alloc.delete_after(before, ptr, n)} -> event;
};


template<class A, class E, class P, class N>
concept has_delete_after_free_function = requires(A alloc, E before, P ptr, N n)
{
  {delete_after(alloc, before, ptr, n)} -> event;
};


// this is the type of delete_after
struct dispatch_delete_after
{
  // this dispatch path calls the member function
  template<class Allocator, class Event, class P, class N>
    requires has_delete_after_member_function<Allocator&&, Event&&, P&&, N&&>
  constexpr auto operator()(Allocator&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    return std::forward<Allocator>(alloc).delete_after(std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Allocator, class Event, class P, class N>
    requires (!has_delete_after_member_function<Allocator&&, Event&&, P&&, N&&> and
               has_delete_after_free_function<Allocator&&, Event&&, P&&, N&&>)
  constexpr auto operator()(Allocator&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    return delete_after(std::forward<Allocator>(alloc), std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // the default path
  //   1. calls destroy_after
  //   2. calls deallocate_after
  //
  //   XXX P needs to be allocator_traits<A>::pointer
  //   XXX N should be allocator_traits<A>::size_type
  template<asynchronous_allocator A, event E, class P, class N>
    requires (!has_delete_after_member_function<A&&, E&&, P, N> and
              !has_delete_after_free_function<A&&, E&&, P, N> and
              executor_associate<A&&>)
  constexpr auto operator()(A&& alloc, E&& before, P ptr, N n) const
  {
    // destroy
    auto after_destructors = ubu::destroy_after(std::forward<A>(alloc), std::forward<E>(before), ptr, n);

    // deallocate
    return deallocate_after(std::forward<A>(alloc), std::move(after_destructors), ptr, n);
  }

};


} // end detail


namespace
{

constexpr detail::dispatch_delete_after delete_after;

} // end anonymous namespace


template<class A, class E, class P, class N>
using delete_after_result_t = decltype(ubu::delete_after(std::declval<A>(), std::declval<E>(), std::declval<P>(), std::declval<N>()));


} // end ubu

#include "../../detail/epilogue.hpp"

