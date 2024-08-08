#pragma once

#include "../../../detail/prologue.hpp"

#include "concepts/asynchronous_view_of.hpp"
#include "detail/custom_allocate_after.hpp"
#include "detail/one_extending_default_allocate_after.hpp"
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
  constexpr asynchronous_view_of<T,S> auto operator()(A&& alloc, B&& before, S&& shape) const
  {
    return custom_allocate_after<T>(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape));
  }

  // this dispatch path calls one_extending_default_allocate_after
  template<class A, happening B, coordinate S>
    requires (not has_custom_allocate_after<T, A&&, B&&, const S&>
              and has_one_extending_default_allocate_after<T,A&&,B&&,const S&>)
  constexpr asynchronous_view_of<T,S> auto operator()(A&& alloc, B&& before, const S& shape) const
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

