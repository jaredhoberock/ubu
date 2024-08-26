#pragma once

#include "../../../detail/prologue.hpp"

#include "detail/custom_deallocate.hpp"
#include "detail/decomposing_default_deallocate.hpp"
#include <utility>

namespace ubu
{
namespace detail
{


// this is the type of deallocate
struct dispatch_deallocate
{
  // this dispatch path calls the allocator's customization of deallocate
  template<class A, class V>
    requires has_custom_deallocate<A&&,V&&>
  constexpr void operator()(A&& alloc, V&& view) const
  {
    custom_deallocate(std::forward<A>(alloc), std::forward<V>(view));
  }

  // this dispatch path calls decomposing_default_deallocate
  template<class A, view V>
    requires (not has_custom_deallocate<A&&, V>
              and has_decomposing_default_deallocate<A&&, V>)
  constexpr void operator()(A&& alloc, V tensor) const
  {
    decomposing_default_deallocate_after(std::forward<A>(alloc), tensor);
  }
};


} // end detail


inline constexpr detail::dispatch_deallocate deallocate;


} // end ubu

#include "../../../detail/epilogue.hpp"

