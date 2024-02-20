#pragma once

#include "../../../detail/prologue.hpp"

#include "../concepts/integral_like.hpp"
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
              and integral_like<T>)
  constexpr std::size_t operator()() const
  {
    return 1;
  }

  template<class T>
    requires (not has_rank_static_member_variable<std::remove_cvref_t<T>>
              and not has_rank_static_member_function<std::remove_cvref_t<T>>
              and not integral_like<T>
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


template<tuple_like T, std::size_t... I>
constexpr bool is_tuple_like_of_types_each_with_static_rank_impl(std::index_sequence<I...>)
{
  // check that each element type may be used with the rank CPO
  return (... and rank_impl::detected<rank_result_t, std::tuple_element_t<I,T>>);
}

template<tuple_like T>
struct is_tuple_like_of_types_each_with_static_rank<T>
{
  static constexpr bool value = is_tuple_like_of_types_each_with_static_rank_impl<T>(tuple_indices<T>);
};


} // end detail
} // end ubu


#include "../../../detail/epilogue.hpp"

