#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "rebind_allocator.hpp"
#include <memory>
#include <type_traits>
#include <utility>

namespace ubu
{

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


template<class A, class E, class P, class N>
concept has_finally_deallocate_after_customization = has_finally_deallocate_after_member_function<A,E,P,N> or has_finally_deallocate_after_free_function<A,E,P,N>;


// this is the type of finally_deallocate_after
struct dispatch_finally_deallocate_after
{
  // this dispatch path calls the member function
  template<class Allocator, class Event, class P, class N>
    requires has_finally_deallocate_after_member_function<Allocator&&, Event&&, P&&, N&&>
  constexpr auto operator()(Allocator&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    return std::forward<Allocator>(alloc).finally_deallocate_after(std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Allocator, class Event, class P, class N>
    requires (!has_finally_deallocate_after_member_function<Allocator&&, Event&&, P&&, N&&> and
               has_finally_deallocate_after_free_function<Allocator&&, Event&&, P&&, N&&>)
  constexpr auto operator()(Allocator&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    return finally_deallocate_after(std::forward<Allocator>(alloc), std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // this dispatch path tries to rebind and then call finally_deallocate_after again
  template<class Allocator, class Event, class P, class N>
    requires (!has_finally_deallocate_after_member_function<Allocator&&, Event&&, P&&, N&&> and
              !has_finally_deallocate_after_free_function<Allocator&&, Event&&, P&&, N&&> and
              has_finally_deallocate_after_customization<
                rebind_allocator_result_t<typename std::pointer_traits<std::remove_cvref_t<P>>::element_type,Allocator&&>,
                Event&&, P&&, N&&
              >)
  constexpr void operator()(Allocator&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    auto rebound_alloc = rebind_allocator<typename std::pointer_traits<std::remove_cvref_t<P>>::element_type>(std::forward<Allocator>(alloc));
    return (*this)(rebound_alloc, std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // the default path calls deallocate_after
  template<class Allocator, class Event, class P, class N>
    requires (!has_finally_deallocate_after_member_function<Allocator&&, Event&&, P&&, N&&> and
              !has_finally_deallocate_after_free_function<Allocator&&, Event&&, P&&, N&&> and
              !has_finally_deallocate_after_customization<
                rebind_allocator_result_t<typename std::pointer_traits<std::remove_cvref_t<P>>::element_type,Allocator&&>,
                Event&&, P&&, N&&
              >)
  constexpr decltype(auto) operator()(Allocator&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    return deallocate_after(std::forward<Allocator>(alloc), std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_finally_deallocate_after finally_deallocate_after;

} // end anonymous namespace


template<class A, class E, class P, class N>
using finally_deallocate_after_result_t = decltype(ubu::finally_deallocate_after(std::declval<A>(), std::declval<E>(), std::declval<P>(), std::declval<N>()));


} // end ubu

#include "../../detail/epilogue.hpp"

