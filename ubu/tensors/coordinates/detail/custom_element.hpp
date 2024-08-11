#pragma once

#include "../../../detail/prologue.hpp"

#include <type_traits>
#include <utility>

namespace ubu::detail
{


template<class T>
concept not_void = not std::is_void_v<T>;

template<class T, class C>
concept has_element_member_function = requires(T obj, C coord)
{
  { obj.element(coord) } -> not_void;
};

template<class T, class C>
concept has_element_free_function = requires(T obj, C coord)
{
  { element(obj,coord) } -> not_void;
};

template<class T, class C>
concept has_operator_bracket = requires(T obj, C coord)
{
  { obj[coord] } -> not_void;
};

template<class T, class C>
concept has_operator_parens = requires(T obj, C coord)
{
  { obj(coord) } -> not_void;
};


template<class T, class C>
concept has_custom_element =
  has_element_member_function<T,C>
  or has_element_free_function<T,C>
  or has_operator_bracket<T,C>
  or has_operator_parens<T,C>
;


template<class T, class C>
  requires has_custom_element<T,C>
constexpr decltype(auto) custom_element(T&& obj, C&& coord)
{
  if constexpr(has_element_member_function<T&&,C&&>)
  {
    return std::forward<T>(obj).element(std::forward<C>(coord));
  }
  else if constexpr(has_element_free_function<T&&,C&&>)
  {
    return element(std::forward<T>(obj), std::forward<C>(coord));
  }
  else if constexpr(has_operator_bracket<T&&,C&&>)
  {
    return std::forward<T>(obj)[std::forward<C>(coord)];
  }
  else
  {
    return std::forward<T>(obj)(std::forward<C>(coord));
  }
}

template<class T, class C>
using custom_element_result_t = decltype(custom_element(std::declval<T>(), std::declval<C>()));


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

