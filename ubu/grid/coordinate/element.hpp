#pragma once


#include "../../detail/prologue.hpp"

#include <type_traits>
#include <utility>

namespace ubu::detail
{

template<class T>
concept new_not_void = not std::is_void_v<T>;

template<class T, class C>
concept has_element_member_function = requires(T obj, C coord)
{
  { obj.element(coord) } -> new_not_void;
};

template<class T, class C>
concept has_element_free_function = requires(T obj, C coord)
{
  { element(obj,coord) } -> new_not_void;
};

template<class T, class C>
concept has_operator_bracket = requires(T obj, C coord)
{
  { obj[coord] } -> new_not_void;
};

template<class T, class C>
concept has_operator_parens = requires(T obj, C coord)
{
  { obj(coord) } -> new_not_void;
};


struct dispatch_element
{
  // try obj.element(coord)
  template<class T, class C>
    requires has_element_member_function<T&&,C&&>
  constexpr decltype(auto) operator()(T&& obj, C&& coord) const
  {
    return std::forward<T>(obj).element(std::forward<C>(coord));
  }

  // else, try element(obj,coord)
  template<class T, class C>
    requires(not has_element_member_function<T&&,C&&>
             and has_element_free_function<T&&,C&&>)
  constexpr decltype(auto) operator()(T&& obj, C&& coord) const
  {
    return element(std::forward<T>(obj), std::forward<C>(coord));
  }

  // else, try obj[coord]
  template<class T, class C>
    requires(not has_element_member_function<T&&,C&&>
             and not has_element_free_function<T&&,C&&>
             and has_operator_bracket<T&&,C&&>)
  constexpr decltype(auto) operator()(T&& obj, C&& coord) const
  {
    return std::forward<T>(obj)[std::forward<C>(coord)];
  }

  // else, try obj(coord)
  template<class T, class C>
    requires(not has_element_member_function<T&&,C&&>
             and not has_element_free_function<T&&,C&&>
             and not has_operator_bracket<T&&,C&&>
             and has_operator_parens<T&&,C&&>)
  constexpr decltype(auto) operator()(T&& obj, C&& coord) const
  {
    return std::forward<T>(obj)(std::forward<C>(coord));
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

