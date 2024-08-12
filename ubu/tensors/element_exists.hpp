#pragma once

#include "../detail/prologue.hpp"

#include "../utilities/integrals/size.hpp"
#include "detail/recursive_element_exists.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

struct dispatch_element_exists
{
  template<class T, class C>
    requires has_recursive_element_exists<T&&,C&&>
  constexpr bool operator()(T&& arg, C&& coord) const
  {
    return detail::recursive_element_exists(std::forward<T>(arg), std::forward<C>(coord));
  }
};

} // end detail

inline constexpr detail::dispatch_element_exists element_exists;

} // end ubu

#include "../detail/epilogue.hpp"

