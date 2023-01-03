#pragma once


#include "../../detail/prologue.hpp"

#include "detail/number.hpp"
#include <type_traits>
#include <utility>


namespace ubu::detail
{


template<std::size_t i, class T>
concept has_element_member_function = requires(T obj) { obj.template element<i>(); };

template<std::size_t i, class T>
concept has_element_free_function = requires(T obj) { element<i>(obj); };

template<class T, class Arg>
concept has_operator_bracket = requires(T obj, Arg arg) { obj[arg]; };

template<std::size_t i, class T>
concept has_get = requires(T obj) { get<i>(obj); };


template<std::size_t i>
struct dispatch_element
{
  // when T has a member function template arg.element<i>(), return that
  template<class T>
    requires has_element_member_function<i,T&&>
  constexpr auto operator()(T&& arg) const
    -> decltype(std::forward<T>(arg).template element<i>())
  {
    return std::forward<T>(arg).template element<i>();
  }

  // else, when T has a free function template element<i>(arg), return that
  template<class T>
    requires(!has_element_member_function<i,T&&> and
              has_element_free_function<i,T&&>)
  constexpr auto operator()(T&& arg) const
    -> decltype(element<i>(std::forward<T>(arg)))
  {
    return element<i>(std::forward<T>(arg));
  }

  // else, when T is just a number and the index i is zero, just return the number
  template<class T>
    requires(!has_element_free_function<i,T&&> and 
             !has_element_free_function<i,T&&> and
             number<std::remove_cvref_t<T&&>> and
             i == 0)
  constexpr T&& operator()(T&& num) const
  {
    return std::forward<T>(num);
  }

  // else, when T has get<i>(arg), return that
  template<class T>
    requires(!has_element_free_function<i,T&&> and 
             !has_element_free_function<i,T&&> and
             !number<T&&> and
             has_get<i,T&&>)
  constexpr auto operator()(T&& arg) const
    -> decltype(get<i>(std::forward<T>(arg)))
  {
    return get<i>(std::forward<T>(arg));
  }

  // else, when T has operator bracket obj[i], return that
  template<class T>
    requires(!has_element_free_function<i,T&&> and 
             !has_element_free_function<i,T&&> and
             !number<T&&> and
             !has_get<i,T&&> and
             has_operator_bracket<T&&,std::size_t>)
  constexpr auto operator()(T&& arg) const
    -> decltype(std::forward<T>(arg)[i])
  {
    return std::forward<T>(arg)[i];
  }
};


} // end ubu::detail


namespace ubu
{


namespace
{

template<std::size_t i>
constexpr detail::dispatch_element<i> element;

} // end anonymous namespace


template<std::size_t i, class T>
using element_t = std::remove_cvref_t<decltype(ubu::element<i>(std::declval<T>()))>;


} // end ubu


#include "../../detail/epilogue.hpp"

