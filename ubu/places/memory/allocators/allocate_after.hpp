#pragma once

#include "../../../detail/prologue.hpp"

#include "concepts/asynchronous_allocation.hpp"
#include "detail/custom_allocate_after.hpp"
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

  // this dispatch path rebinds the allocator and recurses
  template<class A, class B, class S>
    requires (not has_custom_allocate_after<T, A&&, B&&, S&&>
              and has_rebind_allocator<T, A&&>
              and has_custom_allocate_after<
                T, rebind_allocator_result_t<T,A&&>, B&&, S&&
              >)
  constexpr auto operator()(A&& alloc, B&& before, S&& shape) const
  {
    auto rebound_alloc = rebind_allocator<T>(std::forward<A>(alloc));

    // XXX there's not a real need to recurse, we should simply call custom_allocate_after directly
    return (*this)(rebound_alloc, std::forward<B>(before), std::forward<S>(shape));
  }
};


} // end detail


template<class T>
inline constexpr detail::dispatch_allocate_after<T> allocate_after;


template<class T, class A, class B, class S>
using allocate_after_result_t = decltype(allocate_after<T>(std::declval<A>(), std::declval<B>(), std::declval<S>()));


} // end ubu


#include "../../../detail/epilogue.hpp"

