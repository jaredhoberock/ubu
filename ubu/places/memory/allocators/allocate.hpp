#pragma once

#include "../../../detail/prologue.hpp"

#include "detail/custom_allocate.hpp"
#include <utility>

namespace ubu
{
namespace detail
{


// this is the type of allocate
template<class T>
struct dispatch_allocate
{
  // this dispatch path calls the allocator's customization of allocate
  template<class A, class S>
    requires has_custom_allocate<T,A&&,S&&>
  constexpr memory_view_of<T,S> auto operator()(A&& alloc, S&& shape) const
  {
    return custom_allocate<T>(std::forward<A>(alloc), std::forward<S>(shape));
  }

  // XXX another dispatch path should call one_extending_default_allocate
};


} // end detail


template<class T>
inline constexpr detail::dispatch_allocate<T> allocate;


template<class T, class A, class S>
using allocate_result_t = decltype(ubu::allocate<T>(std::declval<A>(), std::declval<S>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

