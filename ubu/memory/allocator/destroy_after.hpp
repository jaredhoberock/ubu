#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "../../execution/executor/dependent_on.hpp"
#include "../../execution/executor/execute_after.hpp"
#include "../pointer.hpp"
#include "destroy.hpp"
#include "traits/allocator_size_t.hpp"
#include "traits/allocator_value_t.hpp"
#include <utility>


namespace ubu
{

namespace detail
{


template<class A, class E, class H, class P, class N>
concept has_destroy_after_member_function = requires(A alloc, E exec, H before, P ptr, N n)
{
  alloc.destroy_after(exec, before, ptr, n);
};


template<class A, class E, class H, class P, class N>
concept has_destroy_after_free_function = requires(A alloc, E exec, H before, P ptr, N n)
{
  destroy_after(alloc, exec, before, ptr, n);
};


// this is the type of destroy_after
struct dispatch_destroy_after
{
  // this dispatch path calls the member function
  template<class Alloc, class Exec, class Happening, class P, class N>
    requires has_destroy_after_member_function<Alloc&&, Exec&&, Happening&&, P&&, N&&>
  constexpr auto operator()(Alloc&& alloc, Exec&& exec, Happening&& before, P&& ptr, N&& n) const
  {
    return std::forward<Alloc>(alloc).destroy_after(std::forward<Exec>(exec), std::forward<Happening>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Alloc, class Exec, class Happening, class P, class N>
    requires (!has_destroy_after_member_function<Alloc&&, Exec&&, Happening&&, P&&, N&&> and
               has_destroy_after_free_function<Alloc&&, Exec&&, Happening&&, P&&, N&&>)
  constexpr auto operator()(Alloc&& alloc, Exec&& exec, Happening&& before, P&& ptr, N&& n) const
  {
    return destroy_after(std::forward<Alloc>(alloc), std::forward<Exec>(exec), std::forward<Happening>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // path for objects with destructors
  template<pointer_like P, allocator_of<pointer_pointee_t<P>> A, executor E, happening H>
    requires (!has_destroy_after_member_function<A&&, E&&, H&&, P&&, allocator_size_t<A>> and
              !has_destroy_after_free_function<A&&, E&&, H&&, P&&, allocator_size_t<A>> and
              !std::is_trivially_destructible_v<pointer_pointee_t<P>>
             )
  constexpr auto operator()(A&& alloc, E&& exec, H&& before, P ptr, allocator_size_t<A> n) const
  {
    // execute destructors
    // XXX this needs to be a bulk_execute call
    return execute_after(std::forward<E>(exec), std::forward<H>(before), [=]
    {
      for(allocator_size_t<A> i = 0; i < n; ++i)
      {
        destroy(alloc, ptr + i);
      }
    });
  }

  // path for objects without destructors
  template<pointer_like P, allocator_of<pointer_pointee_t<P>> A, executor E, happening H>
    requires (!has_destroy_after_member_function<A&&, E&&, H&&, P&&, allocator_size_t<A>> and
              !has_destroy_after_free_function<A&&, E&&, H&&, P&&, allocator_size_t<A>> and
              std::is_trivially_destructible_v<pointer_pointee_t<P>>
             )
  constexpr auto operator()(A&& alloc, E&& exec, H&& before, P ptr, allocator_size_t<A> n) const
  {
    return dependent_on(std::forward<E>(exec), std::forward<H>(before));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_destroy_after destroy_after;

} // end anonymous namespace


template<class A, class E, class H, class P, class N>
using destroy_after_result_t = decltype(ubu::destroy_after(std::declval<A>(), std::declval<E>(), std::declval<H>(), std::declval<P>(), std::declval<N>()));


} // end ubu

#include "../../detail/epilogue.hpp"

