#pragma once

#include "../detail/prologue.hpp"
#include "../utilities/integrals/size.hpp"
#include <concepts>
#include <ranges>
#include <utility>

namespace ubu
{
namespace detail
{

template<class T, class C>
concept has_element_exists_member_function = requires(T arg, C coord)
{
  { arg.element_exists(coord) } -> std::convertible_to<bool>;
};

template<class T, class C>
concept has_element_exists_free_function = requires(T arg, C coord)
{
  { element_exists(arg, coord) } -> std::convertible_to<bool>;
};

struct dispatch_element_exists
{
  template<class T, class C>
  constexpr bool operator()(T&& arg, C&& coord) const
  {
    if constexpr (has_element_exists_member_function<T&&,C&&>)
    {
      return std::forward<T>(arg).element_exists(std::forward<C>(coord));
    }
    else if constexpr (has_element_exists_free_function<T&&,C&&>)
    {
      return element_exists(std::forward<T>(arg), std::forward<C>(coord));
    }
    else if constexpr (sized<T&&>)
    {
      return true;
    }
    else
    {
      static_assert(sized<T&&>, "Couldn't find customization for element_exists.");
    }
  }
};

} // end detail

inline constexpr detail::dispatch_element_exists element_exists;

} // end ubu

#include "../detail/epilogue.hpp"

