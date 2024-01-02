#pragma once

#include "../detail/prologue.hpp"
#include "size.hpp"
#include <concepts>
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
concept has_size = requires(T arg)
{
  size(arg);
};


struct dispatch_element_exists
{
  template<class T, class C>
    requires has_element_exists_member_function<T&&,C&&>
  constexpr bool operator()(T&& arg, C&& coord) const
  {
    return std::forward<T>(arg).element_exists(std::forward<C>(coord));
  }

  template<class T, class C>
    requires (not has_element_exists_member_function<T&&,C&&>
              and has_element_exists_free_function<T&&,C&&>)
  constexpr bool operator()(T&& arg, C&& coord) const
  {
    return element_exists(std::forward<T>(arg), std::forward<T>(coord));
  }

  template<class T, class C>
    requires (not has_element_exists_member_function<T&&,C&&>
              and not has_element_exists_free_function<T&&,C&&>
              and has_size<T&&>)
  constexpr bool operator()(T&&, C&&) const
  {
    return true;
  }
};

} // end detail

inline constexpr detail::dispatch_element_exists element_exists;

} // end ubu

#include "../detail/epilogue.hpp"

