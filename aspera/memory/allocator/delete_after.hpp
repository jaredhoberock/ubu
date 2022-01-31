#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "../allocator/asynchronous_deallocator.hpp"
#include "../allocator/deallocate_after.hpp"
#include "../allocator/destroy_after.hpp"
#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class D, class E, class P, class N>
concept has_delete_after_member_function = requires(D deleter, E before, P ptr, N n)
{
  {deleter.delete_after(before, ptr, n)} -> event;
};


template<class D, class E, class P, class N>
concept has_delete_after_free_function = requires(D deleter, E before, P ptr, N n)
{
  {delete_after(deleter, before, ptr, n)} -> event;
};


// this is the type of delete_after
struct dispatch_delete_after
{
  // this dispatch path calls the member function
  template<class Deleter, class Event, class P, class N>
    requires has_delete_after_member_function<Deleter&&, Event&&, P&&, N&&>
  constexpr auto operator()(Deleter&& deleter, Event&& before, P&& ptr, N&& n) const
  {
    return std::forward<Deleter>(deleter).delete_after(std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Deleter, class Event, class P, class N>
    requires (!has_delete_after_member_function<Deleter&&, Event&&, P&&, N&&> and
               has_delete_after_free_function<Deleter&&, Event&&, P&&, N&&>)
  constexpr auto operator()(Deleter&& deleter, Event&& before, P&& ptr, N&& n) const
  {
    return delete_after(std::forward<Deleter>(deleter), std::forward<Event>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // the default path
  //   1. calls destroy_after
  //   2. calls deallocate_after
  //
  //   XXX P needs to be allocator_traits<A>::pointer
  //   XXX N should be allocator_traits<A>::size_type
  template<asynchronous_deallocator A, event E, class P, class N>
    requires (!has_delete_after_member_function<A&&, E&&, P, N> and
              !has_delete_after_free_function<A&&, E&&, P, N> and
              executor_associate<A&&>)
  constexpr auto operator()(A&& alloc, E&& before, P ptr, N n) const
  {
    // destroy
    auto after_destructors = ASPERA_NAMESPACE::destroy_after(std::forward<A>(alloc), std::forward<E>(before), ptr, n);

    // deallocate
    return ASPERA_NAMESPACE::deallocate_after(std::forward<A>(alloc), std::move(after_destructors), ptr, n);
  }

};


} // end detail


namespace
{

constexpr detail::dispatch_delete_after delete_after;

} // end anonymous namespace


template<class D, class E, class P, class N>
using delete_after_result_t = decltype(ASPERA_NAMESPACE::delete_after(std::declval<D>(), std::declval<E>(), std::declval<P>(), std::declval<N>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"


