#pragma once

#include "../detail/prologue.hpp"

#include "detail/number.hpp"
#include <array>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
concept has_size_static_member_function = requires
{
  {T::size()} -> std::convertible_to<std::size_t>;
};

template<class T>
concept has_size_member_function = requires(T obj)
{
  {obj.size()} -> std::convertible_to<std::size_t>;
};

template<class T>
concept has_size_free_function = requires(T obj)
{
  {size(obj)} -> std::convertible_to<std::size_t>;
};


template<class T>
struct is_tuple_of_types_each_with_static_size;


struct dispatch_size
{
  // static cases do not take a parameter
  template<class T>
    requires has_size_static_member_function<std::remove_cvref_t<T>>
  constexpr std::size_t operator()() const
  {
    return std::remove_cvref_t<T>::size();
  }

  template<class T>
    requires (!has_size_static_member_function<std::remove_cvref_t<T>> and
              number<std::remove_cvref_t<T>>)
  constexpr std::size_t operator()() const
  {
    return 1;
  }

  template<class T>
    requires (!has_size_static_member_function<std::remove_cvref_t<T>> and
              !number<T> and
              is_tuple_of_types_each_with_static_size<std::remove_cvref_t<T>>::value)
  constexpr std::size_t operator()() const
  {
    return std::tuple_size<std::remove_cvref_t<T>>::value;
  }
           

  // dynamic cases do take a parameter
  template<class T>
    requires has_size_member_function<T&&>
  constexpr std::size_t operator()(T&& arg) const
  {
    return std::forward<T>(arg).size();
  }

  template<class T>
    requires (!has_size_member_function<T&&> and
               has_size_free_function<T&&>)
  constexpr std::size_t operator()(T&& arg) const
  {
    return size(std::forward<T>(arg));
  }

  template<class T>
    requires (!has_size_member_function<T&&> and
              !has_size_free_function<T&&>)
  constexpr auto operator()(T&&) const
    -> decltype(operator()<std::remove_cvref_t<T&&>>())
  {
    return operator()<std::remove_cvref_t<T&&>>();
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_size size;

} // end anonymous namespace


template<class T>
using size_result_t = decltype(ASPERA_NAMESPACE::size(std::declval<T>()));


namespace detail
{


template<class T>
concept has_static_size = requires
{ 
  size.operator()<T>();
};


} // end detail


template<class T>
  requires detail::has_static_size<T>
static constexpr auto size_v = size.operator()<T>();


namespace detail
{


// XXX this needs to be generalize to all tuple-like types beyond std::tuple and std::pair and std::array
template<class T>
struct is_tuple_of_types_each_with_static_size
{
  static constexpr bool value = false;
};


template<template<class...> class Template, class... Types>
concept detected = requires
{
  typename Template<Types...>;
};


template<class... Types>
struct is_tuple_of_types_each_with_static_size<std::tuple<Types...>>
{
  // check that each of Types... may be used with the size CPO
  static constexpr bool value = (... && detected<size_result_t, Types>);
};

template<class T1, class T2>
struct is_tuple_of_types_each_with_static_size<std::pair<T1,T2>>
{
  // check that T1 & T1 may be used with the size CPO
  static constexpr bool value = detected<size_result_t,T1> and detected<size_result_t,T2>;
};

template<class T, std::size_t N>
struct is_tuple_of_types_each_with_static_size<std::array<T,N>>
{
  // check that T may be used with the size CPO
  static constexpr bool value = detected<size_result_t,T>;
};


} // end detail


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

