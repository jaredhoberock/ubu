#pragma once

#include "../../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "detail/custom_deallocate_after.hpp"
#include <utility>

namespace ubu
{

namespace detail
{


// this is the type of deallocate_after
struct dispatch_deallocate_after
{
  // this dispatch path calls the allocator's customization of deallocate_after
  template<class A, class B, class T>
    requires has_custom_deallocate_after<A&&, B&&, T&&>
  constexpr auto operator()(A&& alloc, B&& before, T&& tensor) const
  {
    return custom_deallocate_after(std::forward<A>(alloc), std::forward<B>(before), std::forward<T>(tensor));
  }

  // XXX add a default dispatch path analogous to one_extending_allocate_after
};


} // end detail


namespace
{

constexpr detail::dispatch_deallocate_after deallocate_after;

} // end anonymous namespace


template<class A, class B, class T>
using deallocate_after_result_t = decltype(deallocate_after(std::declval<A>(), std::declval<B>(), std::declval<T>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

