#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "asynchronous_deallocator.hpp"
#include <memory>
#include <utility>

ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class A, class E, class P, class N>
concept has_finally_deallocate_after_member_function = requires(A alloc, E before, P ptr, N n)
{
  alloc.finally_deallocate_after(before, ptr, n);
};


template<class A, class E, class P, class N>
concept has_finally_deallocate_after_free_function = requires(A alloc, E before, P ptr, N n)
{
  finally_deallocate_after(alloc, before, ptr, n);
};


// this is the type of finally_deallocate_after
struct dispatch_finally_deallocate_after
{
  // this dispatch path calls the member function
  template<class Alloc, class Event, class P, class N>
    requires has_finally_deallocate_after_member_function<Alloc&&, Event&&, P&&, N&&>
  constexpr auto operator()(Alloc&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    return std::forward<Alloc>(alloc).finally_deallocate_after(std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Alloc, class Event, class P, class N>
    requires (!has_finally_deallocate_after_member_function<Alloc&&, Event&&, P&&, N&&> and
               has_finally_deallocate_after_free_function<Alloc&&, Event&&, P&&, N&&>)
  constexpr auto operator()(Alloc&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    return finally_deallocate_after(std::forward<Alloc>(alloc), std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // the default path calls deallocate_after
  template<asynchronous_deallocator Alloc, class Event, class P, class N>
    requires (!has_finally_deallocate_after_member_function<Alloc&&, Event&&, P&&, N&&> and
              !has_finally_deallocate_after_free_function<Alloc&&, Event&&, P&&, N&&>)
  constexpr void operator()(Alloc&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    deallocate_after(std::forward<Alloc>(alloc), std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_finally_deallocate_after finally_deallocate_after;

} // end anonymous namespace


template<class A, class E, class P, class N>
using finally_deallocate_after_result_t = decltype(ASPERA_NAMESPACE::finally_deallocate_after(std::declval<A>(), std::declval<E>(), std::declval<P>(), std::declval<N>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

