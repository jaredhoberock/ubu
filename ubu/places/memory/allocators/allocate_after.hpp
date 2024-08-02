#pragma once

#include "../../../detail/prologue.hpp"

#include "concepts/asynchronous_allocation.hpp"
#include "detail/custom_allocate_after.hpp"
#include "detail/one_extending_default_allocate_after.hpp"
#include "rebind_allocator.hpp"
#include "traits/allocator_value.hpp"
#include <utility>

namespace ubu
{

namespace detail
{


// this is the type of allocate_after
template<class T>
struct dispatch_allocate_after
{
  // this dispatch path calls the allocator's customization of allocate_after
  template<class A, class B, class S>
    requires has_custom_allocate_after<T, A&&, B&&, S&&>
  constexpr asynchronous_allocation auto operator()(A&& alloc, B&& before, S&& shape) const
  {
    return custom_allocate_after<T>(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape));
  }

  // this dispatch path rebinds the allocator and calls the rebound allocator's customization of allocate_after
  // XXX this should be the final fallback path and it should call either custom_allocate_after or one_extending_default_allocate_after
  template<class A, class B, class S>
    requires (not has_custom_allocate_after<T, A&&, B&&, S&&>
              and has_rebind_allocator<T, A&&>
              and has_custom_allocate_after<
                T, rebind_allocator_result_t<T,A&&>, B&&, S&&
              >)
  constexpr asynchronous_allocation auto operator()(A&& alloc, B&& before, S&& shape) const
  {
    auto rebound_alloc = rebind_allocator<T>(std::forward<A>(alloc));

    return custom_allocate_after<T>(rebound_alloc, std::forward<B>(before), std::forward<S>(shape));
  }

  // this dispatch path calls one_extending_default_allocate_after
  // XXX this should be the second fallback path, not the final fallback path
  template<class A, happening B, coordinate S>
    requires (not has_custom_allocate_after<T, A&&, B&&, const S&>
              and not (has_rebind_allocator<T, A&&> and has_custom_allocate_after<T, rebind_allocator_result_t<T,A&&>, B&&, const S&>)
              and has_one_extending_default_allocate_after<T,A&&,B&&,const S&>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, B&& before, const S& shape) const
  {
    return one_extending_default_allocate_after<T>(std::forward<A>(alloc), std::forward<B>(before), shape);
  }
};


} // end detail


template<class T>
inline constexpr detail::dispatch_allocate_after<T> allocate_after;


template<class T, class A, class B, class S>
using allocate_after_result_t = decltype(allocate_after<T>(std::declval<A>(), std::declval<B>(), std::declval<S>()));


} // end ubu


#include "../../../detail/epilogue.hpp"

