#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "../../execution/executor/dependent_on.hpp"
#include "../../execution/executor/execute_after.hpp"
#include "../../execution/executor/executor_associate.hpp"
#include "../pointer.hpp"
#include "destroy.hpp"
#include "traits/allocator_size_t.hpp"
#include "traits/allocator_value_t.hpp"
#include <utility>


namespace ubu
{

namespace detail
{


template<class A, class H, class P, class N>
concept has_destroy_after_member_function = requires(A alloc, H before, P ptr, N n)
{
  alloc.destroy_after(before, ptr, n);
};


template<class A, class H, class P, class N>
concept has_destroy_after_free_function = requires(A alloc, H before, P ptr, N n)
{
  destroy_after(alloc, before, ptr, n);
};


// this is the type of destroy_after
struct dispatch_destroy_after
{
  // this dispatch path calls the member function
  template<class Alloc, class Happening, class P, class N>
    requires has_destroy_after_member_function<Alloc&&, Happening&&, P&&, N&&>
  constexpr auto operator()(Alloc&& alloc, Happening&& before, P&& ptr, N&& n) const
  {
    return std::forward<Alloc>(alloc).destroy_after(std::forward<Happening>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Alloc, class Happening, class P, class N>
    requires (!has_destroy_after_member_function<Alloc&&, Happening&&, P&&, N&&> and
               has_destroy_after_free_function<Alloc&&, Happening&&, P&&, N&&>)
  constexpr auto operator()(Alloc&& alloc, Happening&& before, P&& ptr, N&& n) const
  {
    return destroy_after(std::forward<Alloc>(alloc), std::forward<Happening>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // path for objects with destructors
  template<pointer_like P, allocator_of<pointer_pointee_t<P>> A, happening H>
    requires (!has_destroy_after_member_function<A&&, H&&, P&&, allocator_size_t<A>> and
              !has_destroy_after_free_function<A&&, H&&, P&&, allocator_size_t<A>> and
              executor_associate<A&&> and
              !std::is_trivially_destructible_v<pointer_pointee_t<P>>
             )
  constexpr auto operator()(A&& alloc, H&& before, P ptr, allocator_size_t<A> n) const
  {
    // get an executor
    auto ex = associated_executor(std::forward<A>(alloc));

    // execute destructors
    // XXX this needs to be a bulk_execute call
    return execute_after(ex, std::forward<H>(before), [=]
    {
      for(allocator_size_t<A> i = 0; i < n; ++i)
      {
        destroy(alloc, ptr + i);
      }
    });
  }

  // path for objects without destructors
  template<pointer_like P, allocator_of<pointer_pointee_t<P>> A, happening H>
    requires (!has_destroy_after_member_function<A&&, H&&, P&&, allocator_size_t<A>> and
              !has_destroy_after_free_function<A&&, H&&, P&&, allocator_size_t<A>> and
              executor_associate<A&&> and
              std::is_trivially_destructible_v<pointer_pointee_t<P>>
             )
  constexpr auto operator()(A&& alloc, H&& before, P ptr, allocator_size_t<A> n) const
  {
    // get an executor
    auto ex = associated_executor(std::forward<A>(alloc));

    return dependent_on(ex, std::forward<H>(before));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_destroy_after destroy_after;

} // end anonymous namespace


template<class A, class H, class P, class N>
using destroy_after_result_t = decltype(ubu::destroy_after(std::declval<A>(), std::declval<H>(), std::declval<P>(), std::declval<N>()));


} // end ubu

#include "../../detail/epilogue.hpp"

