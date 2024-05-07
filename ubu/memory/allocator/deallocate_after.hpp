#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "../../tensor/traits/tensor_element.hpp"
#include "../../tensor/vector/span_like.hpp"
#include "rebind_allocator.hpp"
#include <type_traits>
#include <utility>

namespace ubu
{

namespace detail
{


template<class A, class B, class S>
concept has_deallocate_after_member_function = requires(A alloc, B before, S span)
{
  {alloc.deallocate_after(before, span)} -> happening;
};


template<class A, class B, class S>
concept has_deallocate_after_free_function = requires(A alloc, B before, S span)
{
  {deallocate_after(alloc, before, span)} -> happening;
};


template<class A, class B, class S>
concept has_deallocate_after_customization =
  has_deallocate_after_member_function<A,B,S> or
  has_deallocate_after_free_function<A,B,S>
;


// this is the type of deallocate_after
struct dispatch_deallocate_after
{
  // this dispatch path calls the member function
  template<class A, class B, class S>
    requires has_deallocate_after_member_function<A&&, B&&, S&&>
  constexpr auto operator()(A&& alloc, B&& before, S&& span) const
  {
    return std::forward<A>(alloc).deallocate_after(std::forward<B>(before), std::forward<S>(span));
  }

  // this dispatch path calls the free function
  template<class A, class B, class S>
    requires (!has_deallocate_after_member_function<A&&, B&&, S&&> and
               has_deallocate_after_free_function<A&&, B&&, S&&>)
  constexpr auto operator()(A&& alloc, B&& before, S&& span) const
  {
    return deallocate_after(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(span));
  }

  // this dispatch path first does rebind_allocator and then recurses
  template<class A, class B, span_like S>
    requires (not has_deallocate_after_customization<A&&,B&&,S&&> and
              has_rebind_allocator<tensor_element_t<S>,A&&> and
              has_deallocate_after_customization<
                rebind_allocator_result_t<tensor_element_t<S&&>,A&&>,
                B&&, S&&
              >)
  constexpr decltype(auto) operator()(A&& alloc, B&& before, S&& span) const
  {
    auto rebound_alloc = rebind_allocator<tensor_element_t<S>>(std::forward<A>(alloc));
    return (*this)(rebound_alloc, std::forward<B>(before), std::forward<S>(span));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_deallocate_after deallocate_after;

} // end anonymous namespace


template<class A, class B, class S>
using deallocate_after_result_t = decltype(deallocate_after(std::declval<A>(), std::declval<B>(), std::declval<S>()));


} // end ubu

#include "../../detail/epilogue.hpp"

