#pragma once

#include "../detail/prologue.hpp"

#include "detail/number.hpp"
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
concept has_rank_static_member_function = requires
{
  {T::rank()} -> std::convertible_to<std::size_t>;
};

template<class T>
concept has_rank_member_function = requires(T obj)
{
  {obj.rank()} -> std::convertible_to<std::size_t>;
};

template<class T>
concept has_rank_free_function = requires(T obj)
{
  {rank(obj)} -> std::convertible_to<std::size_t>;
};


template<class T>
struct is_tuple_of_types_each_with_static_rank;


struct dispatch_rank
{
  // static cases do not take a parameter
  template<class T>
    requires has_rank_static_member_function<std::remove_cvref_t<T>>
  constexpr std::size_t operator()() const
  {
    return std::remove_cvref_t<T>::rank();
  }

  template<class T>
    requires (!has_rank_static_member_function<std::remove_cvref_t<T>> and
              number<std::remove_cvref_t<T>>)
  constexpr std::size_t operator()() const
  {
    return 0;
  }

  template<class T>
    requires (!has_rank_static_member_function<std::remove_cvref_t<T>> and
              !number<T> and
              is_tuple_of_types_each_with_static_rank<std::remove_cvref_t<T>>::value)
  constexpr std::size_t operator()() const
  {
    return std::tuple_size<std::remove_cvref_t<T>>::value;
  }
           

  // dynamic cases do take a parameter
  template<class T>
    requires has_rank_member_function<T&&>
  constexpr std::size_t operator()(T&& arg) const
  {
    return std::forward<T>(arg).rank();
  }

  template<class T>
    requires (!has_rank_member_function<T&&> and
               has_rank_free_function<T&&>)
  constexpr std::size_t operator()(T&& arg) const
  {
    return rank(std::forward<T>(arg));
  }

  template<class T>
    requires (!has_rank_member_function<T&&> and
              !has_rank_free_function<T&&>)
  constexpr auto operator()(T&&) const
    -> decltype(operator()<std::remove_cvref_t<T&&>>())
  {
    return operator()<std::remove_cvref_t<T&&>>();
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_rank rank;

} // end anonymous namespace


template<class T>
using rank_t = decltype(ASPERA_NAMESPACE::rank(std::declval<T>()));


template<class T>
static constexpr auto rank_v = rank.operator()<T>();


namespace detail
{


// XXX this needs to be generalize to all tuple-like types beyond std::tuple and std::pair
template<class T>
struct is_tuple_of_types_each_with_static_rank
{
  static constexpr bool value = false;
};


template<template<class...> class Template, class... Types>
concept detected = requires
{
  typename Template<Types...>;
};


template<class... Types>
struct is_tuple_of_types_each_with_static_rank<std::tuple<Types...>>
{
  // check that each of Types... may be used with the rank CPO
  static constexpr bool value = (... && detected<rank_t, Types>);
};

template<class T1, class T2>
struct is_tuple_of_types_each_with_static_rank<std::pair<T1,T2>>
{
  // check that T1 & T1 may be used with the rank CPO
  static constexpr bool value = detected<rank_t,T1> and detected<rank_t,T2>;
};


} // end detail


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

