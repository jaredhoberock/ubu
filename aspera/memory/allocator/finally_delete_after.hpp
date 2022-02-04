#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "delete_after.hpp"
#include <utility>

ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class D, class E, class P, class N>
concept has_finally_delete_after_member_function = requires(D deleter, E before, P ptr, N n)
{
  deleter.finally_delete_after(before, ptr, n);
};


template<class D, class E, class P, class N>
concept has_finally_delete_after_free_function = requires(D deleter, E before, P ptr, N n)
{
  finally_delete_after(deleter, before, ptr, n);
};


// this is the type of finally_delete_after
struct dispatch_finally_delete_after
{
  // this dispatch path calls the member function
  template<class Deleter, class Event, class P, class N>
    requires has_finally_delete_after_member_function<Deleter&&, Event&&, P&&, N&&>
  constexpr auto operator()(Deleter&& deleter, Event&& before, P&& ptr, N&& n) const
  {
    return std::forward<Deleter>(deleter).finally_delete_after(std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Deleter, class Event, class P, class N>
    requires (!has_finally_delete_after_member_function<Deleter&&, Event&&, P&&, N&&> and
               has_finally_delete_after_free_function<Deleter&&, Event&&, P&&, N&&>)
  constexpr auto operator()(Deleter&& deleter, Event&& before, P&& ptr, N&& n) const
  {
    return finally_delete_after(std::forward<Deleter>(deleter), std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // XXX this needs to require that delete_after is valid
  template<class Deleter, class Event, class P, class N>
    requires (!has_finally_delete_after_member_function<Deleter&&, Event&&, P&&, N&&> and
              !has_finally_delete_after_free_function<Deleter&&, Event&&, P&&, N&&>)
  constexpr auto operator()(Deleter&& deleter, Event&& before, P&& ptr, N&& n) const
  {
    // discard delete_after's result
    delete_after(std::forward<Deleter>(deleter), std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

};


} // end detail


namespace
{

constexpr detail::dispatch_finally_delete_after finally_delete_after;

} // end anonymous namespace


template<class D, class E, class P, class N>
using finally_delete_after_result_t = decltype(ASPERA_NAMESPACE::finally_delete_after(std::declval<D>(), std::declval<E>(), std::declval<P>(), std::declval<N>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

