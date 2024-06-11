#pragma once

#include "../../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "../../execution/executors/concepts/executor.hpp"
#include "../pointers/pointer_like.hpp"
#include "concepts/asynchronous_allocator.hpp"
#include "delete_after.hpp"
#include <utility>

namespace ubu
{

namespace detail
{


template<class A, class E, class H, class P, class N>
concept has_finally_delete_after_member_function = requires(A alloc, E exec, H before, P ptr, N n)
{
  alloc.finally_delete_after(exec, before, ptr, n);
};


template<class A, class E, class H, class P, class N>
concept has_finally_delete_after_free_function = requires(A alloc, E exec, H before, P ptr, N n)
{
  finally_delete_after(alloc, exec, before, ptr, n);
};


// this is the type of finally_delete_after
struct dispatch_finally_delete_after
{
  // this dispatch path calls the member function
  template<class Allocator, class Executor, class Happening, class P, class N>
    requires has_finally_delete_after_member_function<Allocator&&, Executor&&, Happening&&, P&&, N&&>
  constexpr auto operator()(Allocator&& alloc, Executor&& exec, Happening&& before, P&& ptr, N&& n) const
  {
    return std::forward<Allocator>(alloc).finally_delete_after(std::forward<Executor>(exec), std::forward<Happening>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Allocator, class Executor, class Happening, class P, class N>
    requires (!has_finally_delete_after_member_function<Allocator&&, Executor&&, Happening&&, P&&, N&&> and
               has_finally_delete_after_free_function<Allocator&&, Executor&&, Happening&&, P&&, N&&>)
  constexpr auto operator()(Allocator&& alloc, Executor&& exec, Happening&& before, P&& ptr, N&& n) const
  {
    return finally_delete_after(std::forward<Allocator>(alloc), std::forward<Executor>(exec), std::forward<Happening>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // XXX this needs to require that delete_after is valid
  template<pointer_like P, asynchronous_allocator_of<pointer_pointee_t<P>> A, executor E, happening H, class N>
    requires (!has_finally_delete_after_member_function<A&&, E&&, H&&, P, N> and
              !has_finally_delete_after_free_function<A&&, E&&, H&&, P, N>)
  constexpr auto operator()(A&& alloc, E&& exec, H&& before, P ptr, N n) const
  {
    // discard delete_after's result
    delete_after(std::forward<A>(alloc), std::forward<E>(exec), std::forward<H>(before), ptr, n);
  }

};


} // end detail


namespace
{

constexpr detail::dispatch_finally_delete_after finally_delete_after;

} // end anonymous namespace


template<class A, class E, class H, class P, class N>
using finally_delete_after_result_t = decltype(ubu::finally_delete_after(std::declval<A>(), std::declval<E>(), std::declval<H>(), std::declval<P>(), std::declval<N>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

