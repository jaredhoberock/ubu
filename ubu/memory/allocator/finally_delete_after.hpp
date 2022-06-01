#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "delete_after.hpp"
#include <utility>

namespace ubu
{

namespace detail
{


template<class A, class E, class P, class N>
concept has_finally_delete_after_member_function = requires(A alloc, E before, P ptr, N n)
{
  alloc.finally_delete_after(before, ptr, n);
};


template<class A, class E, class P, class N>
concept has_finally_delete_after_free_function = requires(A alloc, E before, P ptr, N n)
{
  finally_delete_after(alloc, before, ptr, n);
};


// this is the type of finally_delete_after
struct dispatch_finally_delete_after
{
  // this dispatch path calls the member function
  template<class Allocator, class Event, class P, class N>
    requires has_finally_delete_after_member_function<Allocator&&, Event&&, P&&, N&&>
  constexpr auto operator()(Allocator&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    return std::forward<Allocator>(alloc).finally_delete_after(std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Allocator, class Event, class P, class N>
    requires (!has_finally_delete_after_member_function<Allocator&&, Event&&, P&&, N&&> and
               has_finally_delete_after_free_function<Allocator&&, Event&&, P&&, N&&>)
  constexpr auto operator()(Allocator&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    return finally_delete_after(std::forward<Allocator>(alloc), std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // XXX this needs to require that delete_after is valid
  template<class Allocator, class Event, class P, class N>
    requires (!has_finally_delete_after_member_function<Allocator&&, Event&&, P&&, N&&> and
              !has_finally_delete_after_free_function<Allocator&&, Event&&, P&&, N&&>)
  constexpr auto operator()(Allocator&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    // discard delete_after's result
    delete_after(std::forward<Allocator>(alloc), std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

};


} // end detail


namespace
{

constexpr detail::dispatch_finally_delete_after finally_delete_after;

} // end anonymous namespace


template<class A, class E, class P, class N>
using finally_delete_after_result_t = decltype(ubu::finally_delete_after(std::declval<A>(), std::declval<E>(), std::declval<P>(), std::declval<N>()));


} // end ubu

#include "../../detail/epilogue.hpp"

