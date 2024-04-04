#pragma once

#include "../detail/prologue.hpp"
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

template<class T>
concept has_std_ranges_size = requires(T arg)
{
  std::ranges::size(arg);
};

template<class T>
concept has_std_size = requires(T arg)
{
  // std::ranges::size rejects a member function .size() which returns ubu::constant, but std::size does not
  std::size(arg);
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
    else if constexpr (has_std_ranges_size<T&&> or has_std_size<T&&>)
    {
      // XXX we probably need to introduce our own CPO named size
      return true;
    }
    else
    {
      static_assert(has_std_size<T&&>, "Couldn't find customization for element_exists.");
    }
  }
};

} // end detail

inline constexpr detail::dispatch_element_exists element_exists;

} // end ubu

#include "../detail/epilogue.hpp"

