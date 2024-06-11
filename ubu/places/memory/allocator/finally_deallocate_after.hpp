#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../tensor/traits/tensor_element.hpp"
#include "../../../tensor/vector/span_like.hpp"
#include "../../causality/happening.hpp"
#include "rebind_allocator.hpp"
#include <utility>

namespace ubu
{

namespace detail
{


template<class A, class B, class S>
concept has_finally_deallocate_after_member_function = requires(A alloc, B before, S span)
{
  alloc.finally_deallocate_after(before, span);
};


template<class A, class B, class S>
concept has_finally_deallocate_after_free_function = requires(A alloc, B before, S span)
{
  finally_deallocate_after(alloc, before, span);
};


template<class A, class B, class S>
concept has_finally_deallocate_after_customization = has_finally_deallocate_after_member_function<A,B,S> or has_finally_deallocate_after_free_function<A,B,S>;


// this is the type of finally_deallocate_after
struct dispatch_finally_deallocate_after
{
  // this dispatch path calls the member function
  template<class A, class B, class S>
    requires has_finally_deallocate_after_member_function<A&&, B&&, S&&>
  constexpr auto operator()(A&& alloc, B&& before, S&& span) const
  {
    return std::forward<A>(alloc).finally_deallocate_after(std::forward<B>(before), std::forward<S>(span));
  }

  // this dispatch path calls the free function
  template<class A, class B, class S>
    requires (!has_finally_deallocate_after_member_function<A&&, B&&, S&&> and
               has_finally_deallocate_after_free_function<A&&, B&&, S&&>)
  constexpr auto operator()(A&& alloc, B&& before, S&& span) const
  {
    return finally_deallocate_after(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(span));
  }

  // this dispatch path tries to rebind and then call finally_deallocate_after again
  template<class A, class B, span_like S>
    requires (!has_finally_deallocate_after_member_function<A&&, B&&, S&&> and
              !has_finally_deallocate_after_free_function<A&&, B&&, S&&> and
              has_finally_deallocate_after_customization<
                rebind_allocator_result_t<tensor_element_t<S&&>,A&&>,
                B&&, S&&
              >)
  constexpr void operator()(A&& alloc, B&& before, S span) const
  {
    auto rebound_alloc = rebind_allocator<tensor_element_t<S>>(std::forward<A>(alloc));
    return (*this)(rebound_alloc, std::forward<B>(before), std::forward<S>(span));
  }

  // the default path calls deallocate_after
  template<class A, class B, span_like S>
    requires (!has_finally_deallocate_after_member_function<A&&, B&&, S&&> and
              !has_finally_deallocate_after_free_function<A&&, B&&, S&&> and
              !has_finally_deallocate_after_customization<
                rebind_allocator_result_t<tensor_element_t<S&&>,A&&>,
                B&&, S&&
              >)
  constexpr decltype(auto) operator()(A&& alloc, B&& before, S&& span) const
  {
    return deallocate_after(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(span));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_finally_deallocate_after finally_deallocate_after;

} // end anonymous namespace


template<class A, class B, class S>
using finally_deallocate_after_result_t = decltype(finally_deallocate_after(std::declval<A>(), std::declval<B>(), std::declval<S>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

