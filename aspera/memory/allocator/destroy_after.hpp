#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "../../executor/contingent_on.hpp"
#include "../../executor/executor_associate.hpp"
#include "../../executor/then_execute.hpp"
#include "destroy.hpp"
#include <memory>
#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE

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

  // the default path
  // XXX A needs to be an allocator
  // XXX P needs to be allocator_traits<A>::pointer
  // XXX N should be allocator_traits<A>::size_type
  //template<allocator A, event E, class P, class N>
  template<class A, event E, class P, class N>
    requires (!has_destroy_after_member_function<A&&, E&&, P&&, N&&> and
              !has_destroy_after_free_function<A&&, E&&, P&&, N&&> and
              executor_associate<A&&>)
  constexpr auto operator()(const A& alloc, E&& before, P ptr, N n) const
  {
    // get an executor
    auto ex = ASPERA_NAMESPACE::associated_executor(alloc);

    if constexpr(not std::is_trivially_destructible_v<typename std::pointer_traits<P>::element_type>)
    {
      // execute destructors
      // XXX this needs to be a bulk_execute call
      return ASPERA_NAMESPACE::then_execute(ex, std::forward<E>(before), [=]
      {
        for(N i = 0; i < n; ++i)
        {
          ASPERA_NAMESPACE::destroy(alloc, ptr + i);
        }
      });
    }

    return ASPERA_NAMESPACE::contingent_on(ex, std::forward<E>(before));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_destroy_after destroy_after;

} // end anonymous namespace


template<class A, class E, class P, class N>
using destroy_after_result_t = decltype(ASPERA_NAMESPACE::destroy_after(std::declval<A>(), std::declval<E>(), std::declval<P>(), std::declval<N>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

