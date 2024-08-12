#pragma once

#include "../../detail/prologue.hpp"

#include "detail/recursive_element.hpp"
#include <utility>

namespace ubu::detail
{


struct dispatch_element
{
  template<class T, class C>
    requires has_recursive_element<T&&,C&&>
  constexpr decltype(auto) operator()(T&& obj, C&& coord) const
  {
    return recursive_element(std::forward<T>(obj), std::forward<C>(coord));
  }
};


} // end ubu::detail


namespace ubu
{

inline namespace cpos
{

inline constexpr detail::dispatch_element element;

} // end cpos


template<class T, class C>
using element_t = std::remove_cvref_t<decltype(element(std::declval<T>(), std::declval<C>()))>;


} // end ubu


#include "../../detail/epilogue.hpp"

