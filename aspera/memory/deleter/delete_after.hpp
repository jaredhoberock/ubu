#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "../../executor/executor_associate.hpp"
#include "../../executor/then_execute.hpp"
#include "../allocator/deallocate_after.hpp"
#include <memory>
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


template<class A>
concept asynchronous_deallocator =
  // allocator<A> and

  requires{typename std::remove_cvref_t<A>::event_type; } and

  event<typename std::remove_cvref_t<A>::event_type> and

  requires(A a, const typename std::remove_cvref_t<A>::event_type& e, typename std::allocator_traits<std::remove_cvref_t<A>>::pointer ptr, std::size_t n)
  {
    {ASPERA_NAMESPACE::deallocate_after(a, e, ptr, n)} -> event;
  }
;


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
  //   1. asks for an executor and destroy the elements in a bulk_execute
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
    // get an executor
    auto ex = ASPERA_NAMESPACE::associated_executor(std::forward<A>(alloc));

    if constexpr(not std::is_trivially_destructible_v<typename std::pointer_traits<P>::element_type>)
    {
      // execute destructors
      // XXX this should be a bulk_execute
      auto after_destructors = ASPERA_NAMESPACE::then_execute(ex, std::forward<E>(before), [=]
      {
        // XXX should copy the allocator in here and call destroy_at
        printf("dispatch_delete_after: Unimplemented\n");
      });

      // deallocate
      return ASPERA_NAMESPACE::deallocate_after(std::forward<A>(alloc), std::move(before), ptr, n);
    }

    // deallocate
    return ASPERA_NAMESPACE::deallocate_after(std::forward<A>(alloc), std::move(before), ptr, n);
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


