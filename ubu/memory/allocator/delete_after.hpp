#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "../../execution/executor/concepts/executor.hpp"
#include "../../tensor/vector/span_like.hpp"
#include "concepts/asynchronous_allocator.hpp"
#include "deallocate_after.hpp"
#include "destroy_after.hpp"
#include <utility>


namespace ubu
{

namespace detail
{


// XXX this should have a path that attempts a rebind


template<class A, class E, class B, class S>
concept has_delete_after_member_function = requires(A alloc, E exec, B before, S span)
{
  {alloc.delete_after(exec, before, span)} -> happening;
};


template<class A, class E, class B, class S>
concept has_delete_after_free_function = requires(A alloc, E exec, B before, S span)
{
  {delete_after(alloc, exec, before, span)} -> happening;
};


// this is the type of delete_after
struct dispatch_delete_after
{
  // this dispatch path calls the member function
  template<class A, class E, class B, class S>
    requires has_delete_after_member_function<A&&, E&&, B&&, S&&>
  constexpr auto operator()(A&& alloc, E&& exec, B&& before, S&& span) const
  {
    return std::forward<A>(alloc).delete_after(std::forward<E>(exec), std::forward<B>(before), std::forward<S>(span));
  }

  // this dispatch path calls the free function
  template<class A, class E, class B, class S>
    requires (!has_delete_after_member_function<A&&, E&&, B&&, S&&> and
               has_delete_after_free_function<A&&, E&&, B&&, S&&>)
  constexpr auto operator()(A&& alloc, E&& exec, B&& before, S&& span) const
  {
    return delete_after(std::forward<A>(alloc), std::forward<E>(exec), std::forward<B>(before), std::forward<S>(span));
  }

  // the default path
  //   1. calls destroy_after
  //   2. calls deallocate_after
  template<span_like S, asynchronous_allocator_of<tensor_element_t<S>> A, executor E, happening B>
    requires (!has_delete_after_member_function<A&&, E&&, B&&, S> and
              !has_delete_after_free_function<A&&, E&&, B&&, S>)
  constexpr auto operator()(A&& alloc, E&& exec, B&& before, S span) const
  {
    // destroy
    auto after_destructors = destroy_after(std::forward<A>(alloc), std::forward<E>(exec), std::forward<B>(before), span);

    // deallocate
    return deallocate_after(std::forward<A>(alloc), std::move(after_destructors), span);
  }

};


} // end detail


namespace
{

constexpr detail::dispatch_delete_after delete_after;

} // end anonymous namespace


template<class A, class H, class P, class N>
using delete_after_result_t = decltype(ubu::delete_after(std::declval<A>(), std::declval<H>(), std::declval<P>(), std::declval<N>()));


} // end ubu

#include "../../detail/epilogue.hpp"

