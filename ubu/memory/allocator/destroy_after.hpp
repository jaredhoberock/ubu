#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/after_all.hpp"
#include "../../causality/happening.hpp"
#include "../../execution/executor/execute_after.hpp"
#include "../../tensor/vector/span_like.hpp"
#include "../../tensor/traits/tensor_element.hpp"
#include "destroy.hpp"
#include <utility>


namespace ubu
{

namespace detail
{


template<class A, class E, class B, class S>
concept has_destroy_after_member_function = requires(A alloc, E exec, B before, S span)
{
  alloc.destroy_after(exec, before, span);
};


template<class A, class E, class B, class S>
concept has_destroy_after_free_function = requires(A alloc, E exec, B before, S span)
{
  destroy_after(alloc, exec, before, span);
};


// this is the type of destroy_after
struct dispatch_destroy_after
{
  // this dispatch path calls the member function
  template<class A, class E, class B, class S>
    requires has_destroy_after_member_function<A&&, E&&, B&&, S&&>
  constexpr auto operator()(A&& alloc, E&& exec, B&& before, S&& span) const
  {
    return std::forward<A>(alloc).destroy_after(std::forward<E>(exec), std::forward<B>(before), std::forward<S>(span));
  }

  // this dispatch path calls the free function
  template<class A, class E, class B, class S>
    requires (!has_destroy_after_member_function<A&&, E&&, B&&, S&&> and
               has_destroy_after_free_function<A&&, E&&, B&&, S&&>)
  constexpr auto operator()(A&& alloc, E&& exec, B&& before, S&& span) const
  {
    return destroy_after(std::forward<A>(alloc), std::forward<E>(exec), std::forward<B>(before), std::forward<S>(span));
  }

  // path for objects with destructors
  template<span_like S, allocator_of<tensor_element_t<S>> A, executor E, happening B>
    requires (!has_destroy_after_member_function<A&&, E&&, B&&, S&&> and
              !has_destroy_after_free_function<A&&, E&&, B&&, S&&> and
              !std::is_trivially_destructible_v<tensor_element_t<S>>
             )
  constexpr auto operator()(A&& alloc, E&& exec, B&& before, S span) const
  {
    // execute destructors
    // XXX this needs to be a bulk_execute call
    return execute_after(std::forward<E>(exec), std::forward<B>(before), [=]
    {
      for(auto& e : span)
      {
        destroy(alloc, &e);
      }
    });
  }

  // path for objects without destructors
  template<span_like S, allocator_of<tensor_element_t<S>> A, executor E, happening B>
    requires (!has_destroy_after_member_function<A&&, E&&, B&&, S> and
              !has_destroy_after_free_function<A&&, E&&, B&&, S> and
              std::is_trivially_destructible_v<tensor_element_t<S>>
             )
  constexpr auto operator()(A&&, E&&, B&& before, S span) const
  {
    return after_all(std::forward<B>(before));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_destroy_after destroy_after;

} // end anonymous namespace


template<class A, class E, class B, class S>
using destroy_after_result_t = decltype(destroy_after(std::declval<A>(), std::declval<E>(), std::declval<B>(), std::declval<S>()));


} // end ubu

#include "../../detail/epilogue.hpp"

