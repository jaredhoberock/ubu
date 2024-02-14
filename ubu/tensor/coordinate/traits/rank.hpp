#pragma once

#include "../../../detail/prologue.hpp"

#include "detail/number.hpp"
#include <array>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ubu
{
namespace detail
{


template<class T>
concept has_rank_static_member_variable = requires
{
  { T::rank } -> std::convertible_to<std::size_t>;
};


template<class T>
concept has_rank_static_member_function = requires
{
  { T::rank() } -> std::convertible_to<std::size_t>;
};

template<class T>
concept has_rank_member_function = requires(T arg)
{
  { arg.rank() } -> std::convertible_to<std::size_t>;
};


template<class T>
concept has_rank_free_function = requires(T arg)
{
  { rank(arg) } -> std::convertible_to<std::size_t>;
};


template<class T>
struct is_tuple_like_of_types_each_with_static_rank;


struct dispatch_rank
{
  // static cases do not take a parameter
  template<class T>
    requires has_rank_static_member_variable<std::remove_cvref_t<T>>
  constexpr std::size_t operator()() const
  {
    return std::remove_cvref_t<T>::rank;
  }

  template<class T>
    requires (not has_rank_static_member_variable<std::remove_cvref_t<T>>
              and has_rank_static_member_function<std::remove_cvref_t<T>>)
  constexpr std::size_t operator()() const
  {
    return std::remove_cvref_t<T>::rank();
  }

  template<class T>
    requires (not has_rank_static_member_variable<std::remove_cvref_t<T>>
              and not has_rank_static_member_function<std::remove_cvref_t<T>>
              and number<std::remove_cvref_t<T>>)
  constexpr std::size_t operator()() const
  {
    return 1;
  }

  template<class T>
    requires (not has_rank_static_member_variable<std::remove_cvref_t<T>>
              and not has_rank_static_member_function<std::remove_cvref_t<T>>
              and not number<std::remove_cvref_t<T>>
              and is_tuple_like_of_types_each_with_static_rank<std::remove_cvref_t<T>>::value)
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
    requires (not has_rank_member_function<T&&>
              and has_rank_free_function<T&&>)
  constexpr std::size_t operator()(T&& arg) const
  {
    return rank(std::forward<T>(arg));
  }

  // this final default case drops the parameter and attempts
  // to recurse by calling rank<T>()
  template<class T>
    requires (not has_rank_member_function<T&&>
              and not has_rank_free_function<T&&>)
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
using rank_result_t = decltype(ubu::rank(std::declval<T>()));


namespace detail
{


template<class T>
concept static_rank = requires
{
  rank.operator()<T>();
};


} // end detail


template<detail::static_rank T>
static constexpr auto rank_v = rank.operator()<T>();


namespace detail
{


// XXX this needs to be generalize to all tuple-like types beyond std::tuple and std::pair and std::array
template<class T>
struct is_tuple_like_of_types_each_with_static_rank
{
  static constexpr bool value = false;
};


namespace rank_impl
{


template<template<class...> class Template, class... Types>
concept detected = requires
{
  typename Template<Types...>;
};


} // end rank_impl


template<class... Types>
struct is_tuple_like_of_types_each_with_static_rank<std::tuple<Types...>>
{
  // check that each of Types... may be used with the size CPO
  static constexpr bool value = (... && rank_impl::detected<rank_result_t, Types>);
};

template<class T1, class T2>
struct is_tuple_like_of_types_each_with_static_rank<std::pair<T1,T2>>
{
  // check that T1 & T1 may be used with the size CPO
  static constexpr bool value = rank_impl::detected<rank_result_t,T1> and rank_impl::detected<rank_result_t,T2>;
};

template<class T, std::size_t N>
struct is_tuple_like_of_types_each_with_static_rank<std::array<T,N>>
{
  // check that T may be used with the size CPO
  static constexpr bool value = rank_impl::detected<rank_result_t,T>;
};


} // end detail
} // end ubu


#include "../../../detail/epilogue.hpp"

