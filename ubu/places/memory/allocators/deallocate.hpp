#pragma once

#include "../../../detail/prologue.hpp"

#include "detail/custom_deallocate.hpp"
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

  // XXX another dispatch path should call decomposing_default_deallocate
};


} // end detail


inline constexpr detail::dispatch_deallocate deallocate;


} // end ubu

#include "../../../detail/epilogue.hpp"

