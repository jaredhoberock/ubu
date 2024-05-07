#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "../../execution/executor/concepts/executor.hpp"
#include "../../tensor/vector/span_like.hpp"
#include "concepts/asynchronous_allocator.hpp"
#include "delete_after.hpp"
#include <utility>

namespace ubu
{

namespace detail
{


template<class A, class E, class H, class S>
concept has_finally_delete_after_member_function = requires(A alloc, E exec, H before, S span)
{
  alloc.finally_delete_after(exec, before, span);
};


template<class A, class E, class H, class S>
concept has_finally_delete_after_free_function = requires(A alloc, E exec, H before, S span)
{
  finally_delete_after(alloc, exec, before, span);
};


// this is the type of finally_delete_after
struct dispatch_finally_delete_after
{
  // this dispatch path calls the member function
  template<class A, class E, class B, class S>
    requires has_finally_delete_after_member_function<A&&, E&&, B&&, S&&>
  constexpr auto operator()(A&& alloc, E&& exec, B&& before, S&& span) const
  {
    return std::forward<A>(alloc).finally_delete_after(std::forward<E>(exec), std::forward<B>(before), std::forward<S>(span));
  }

  // this dispatch path calls the free function
  template<class A, class E, class B, class S>
    requires (!has_finally_delete_after_member_function<A&&, E&&, B&&, S&&> and
               has_finally_delete_after_free_function<A&&, E&&, B&&, S&&>)
  constexpr auto operator()(A&& alloc, E&& exec, B&& before, S&& span) const
  {
    return finally_delete_after(std::forward<A>(alloc), std::forward<E>(exec), std::forward<B>(before), std::forward<S>(span));
  }

  // XXX this needs to require that delete_after is valid
  template<span_like S, asynchronous_allocator_of<tensor_element_t<S>> A, executor E, happening B>
    requires (!has_finally_delete_after_member_function<A&&, E&&, B&&, S> and
              !has_finally_delete_after_free_function<A&&, E&&, B&&, S>)
  constexpr auto operator()(A&& alloc, E&& exec, B&& before, S span) const
  {
    // discard delete_after's result
    delete_after(std::forward<A>(alloc), std::forward<E>(exec), std::forward<B>(before), span);
  }

};


} // end detail


namespace
{

constexpr detail::dispatch_finally_delete_after finally_delete_after;

} // end anonymous namespace


template<class A, class E, class B, class S>
using finally_delete_after_result_t = decltype(ubu::finally_delete_after(std::declval<A>(), std::declval<E>(), std::declval<B>(), std::declval<S>()));


} // end ubu

#include "../../detail/epilogue.hpp"

