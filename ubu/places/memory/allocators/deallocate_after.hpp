#pragma once

#include "../../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "detail/custom_deallocate_after.hpp"
#include "detail/decomposing_default_deallocate_after.hpp"
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
  constexpr happening auto operator()(A&& alloc, B&& before, T&& tensor) const
  {
    return custom_deallocate_after(std::forward<A>(alloc), std::forward<B>(before), std::forward<T>(tensor));
  }

  // this dispatch path calls decomposing_default_deallocate_after
  template<class A, class B, class T>
    requires (not has_custom_deallocate_after<A&&, B&&, T&&>
              and has_decomposing_default_deallocate_after<A&&, B&&, T&&>)
  constexpr happening auto operator()(A&& alloc, B&& before, T&& tensor) const
  {
    return decomposing_default_deallocate_after(std::forward<A>(alloc), std::forward<B>(before), std::forward<T>(tensor));
  }
};


} // end detail


inline constexpr detail::dispatch_deallocate_after deallocate_after;


template<class A, class B, class T>
using deallocate_after_result_t = decltype(deallocate_after(std::declval<A>(), std::declval<B>(), std::declval<T>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

