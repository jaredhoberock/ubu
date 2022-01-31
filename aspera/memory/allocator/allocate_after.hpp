#pragma once

#include "../../detail/prologue.hpp"

#include <memory>
#include <utility>

ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class A, class E, class N>
concept has_allocate_after_member_function = requires(A alloc, E before, N n)
{
  // XXX this should check that the result is a future<pointer>
  alloc.allocate_after(before, n);
};


template<class A, class E, class N>
concept has_allocate_after_free_function = requires(A alloc, E before, N n)
{
  // XXX this should check that the result is a future<pointer>
  allocate_after(alloc, before, n);
};


// this is the type of allocate_after
struct dispatch_allocate_after
{
  // this dispatch path calls the member function
  template<class Alloc, class Event, class N>
    requires has_allocate_after_member_function<Alloc&&, Event&&, N&&>
  constexpr auto operator()(Alloc&& alloc, Event&& before, N&& n) const
  {
    return std::forward<Alloc>(alloc).allocate_after(std::forward<Event>(before), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Alloc, class Event, class N>
    requires (!has_allocate_after_member_function<Alloc&&, Event&&, N&&> and
               has_allocate_after_free_function<Alloc&&, Event&&, N&&>)
  constexpr auto operator()(Alloc&& alloc, Event&& before, N&& n) const
  {
    return allocate_after(std::forward<Alloc>(alloc), std::forward<Event>(before), std::forward<N>(n));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_allocate_after allocate_after;

} // end anonymous namespace


template<class A, class E, class N>
using allocate_after_result_t = decltype(ASPERA_NAMESPACE::allocate_after(std::declval<A>(), std::declval<E>(), std::declval<N>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

