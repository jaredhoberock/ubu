#pragma once


#include "../../detail/prologue.hpp"

#include "detail/number.hpp"
#include <type_traits>
#include <utility>

// XXX it would be more convenient to use the name "element" for dynamically indexing into grids
//     and use something more like "get" for statically accessing the ith element of a coordinate
//
//     but, element<i>(coord) isn't identical to get<i>(coord) because coord is allowed to be std::integral when i==0
//     if we wanted to change that, we would need to decide that the rank of a bare integer is 0 (like CuTe does),
//     instead of 1 


namespace ubu::detail
{


template<std::size_t i, class T>
concept has_element_member_function_template = requires(T obj) { obj.template element<i>(); };

template<std::size_t i, class T>
concept has_element_free_function_template = requires(T obj) { element<i>(obj); };

template<class T, class Arg>
concept has_operator_bracket = requires(T obj, Arg arg) { obj[arg]; };

template<std::size_t i, class T>
concept has_get_free_function_template = requires(T obj) { get<i>(obj); };


template<std::size_t i>
struct dispatch_element
{
  // try member function template arg.element<i>()
  template<class T>
    requires has_element_member_function_template<i,T&&>
  constexpr decltype(auto) operator()(T&& arg) const
  {
    return std::forward<T>(arg).template element<i>();
  }

  // else, try free function template element<i>(arg)
  template<class T>
    requires(!has_element_member_function_template<i,T&&> and
              has_element_free_function_template<i,T&&>)
  constexpr decltype(auto) operator()(T&& arg) const
  {
    return element<i>(std::forward<T>(arg));
  }

  // else, if the index i is zero, try just returning a number
  template<class T>
    requires(!has_element_member_function_template<i,T&&> and 
             !has_element_free_function_template<i,T&&> and
             number<std::remove_cvref_t<T&&>> and
             i == 0)
  constexpr T&& operator()(T&& num) const
  {
    return std::forward<T>(num);
  }

  // else, try free function template get<i>(arg)
  template<class T>
    requires(!has_element_member_function_template<i,T&&> and 
             !has_element_free_function_template<i,T&&> and
             !number<T&&> and
             has_get_free_function_template<i,T&&>)
  constexpr decltype(auto) operator()(T&& arg) const
  {
    return get<i>(std::forward<T>(arg));
  }

  // else, try arg[i]
  template<class T>
    requires(!has_element_member_function_template<i,T&&> and 
             !has_element_free_function_template<i,T&&> and
             !number<T&&> and
             !has_get_free_function_template<i,T&&> and
             has_operator_bracket<T&&,std::size_t>)
  constexpr decltype(auto) operator()(T&& arg) const
  {
    return std::forward<T>(arg)[i];
  }
};


} // end ubu::detail


namespace ubu
{


template<std::size_t i>
inline constexpr detail::dispatch_element<i> element;


template<std::size_t i, class T>
using element_t = std::remove_cvref_t<decltype(ubu::element<i>(std::declval<T>()))>;


} // end ubu


#include "../../detail/epilogue.hpp"

