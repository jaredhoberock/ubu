#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "../pointer.hpp"
#include "rebind_allocator.hpp"
#include <type_traits>
#include <utility>

namespace ubu
{

namespace detail
{


template<class A, class E, class P, class N>
concept has_deallocate_after_member_function = requires(A alloc, E before, P ptr, N n)
{
  {alloc.deallocate_after(before, ptr, n)} -> happening;
};


template<class A, class E, class P, class N>
concept has_deallocate_after_free_function = requires(A alloc, E before, P ptr, N n)
{
  {deallocate_after(alloc, before, ptr, n)} -> happening;
};


// this is the type of deallocate_after
struct dispatch_deallocate_after
{
  // this dispatch path calls the member function
  template<class Alloc, class Event, class P, class N>
    requires has_deallocate_after_member_function<Alloc&&, Event&&, P&&, N&&>
  constexpr auto operator()(Alloc&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    return std::forward<Alloc>(alloc).deallocate_after(std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Alloc, class Event, class P, class N>
    requires (!has_deallocate_after_member_function<Alloc&&, Event&&, P&&, N&&> and
               has_deallocate_after_free_function<Alloc&&, Event&&, P&&, N&&>)
  constexpr auto operator()(Alloc&& alloc, Event&& before, P&& ptr, N&& n) const
  {
    return deallocate_after(std::forward<Alloc>(alloc), std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // this dispatch path attempts to first rebind_allocator and then recurse
  template<class Alloc, class Event, pointer_like P, class N>
    requires (!has_deallocate_after_member_function<Alloc&&, Event&&, P, N&&> and
              !has_deallocate_after_free_function<Alloc&&, Event&&, P, N&&> and
              has_rebind_allocator<pointer_pointee_t<P>,Alloc&&>)
  constexpr decltype(auto) operator()(Alloc&& alloc, Event&& before, P ptr, N&& n) const
  {
    auto rebound_alloc = rebind_allocator<pointer_pointee_t<P>>(std::forward<Alloc>(alloc));
    return (*this)(rebound_alloc, std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_deallocate_after deallocate_after;

} // end anonymous namespace


template<class A, class E, class P, class N>
using deallocate_after_result_t = decltype(ubu::deallocate_after(std::declval<A>(), std::declval<E>(), std::declval<P>(), std::declval<N>()));


} // end ubu

#include "../../detail/epilogue.hpp"

