#pragma once

#include "../../../detail/prologue.hpp"

#include <array>
#include <concepts>
#include <functional>
#include <iostream>
#include <tuple>
#include <utility>


namespace ubu::detail
{


template<class T, std::size_t N>
concept has_tuple_element =
  requires(T t)
  {
    typename std::tuple_element_t<N, std::remove_const_t<T>>;

    // XXX WAR circle bug:
    // https://godbolt.org/z/McoW5ez6o
    //
    { get<N>(t) } -> std::convertible_to<const std::tuple_element_t<N, T>&>;
  }
;


template<class T, std::size_t... I>
constexpr bool has_tuple_elements(std::index_sequence<I...>)
{
  return (... and has_tuple_element<T,I>);
}


template<class T>
concept tuple_like_impl =
  not std::is_reference_v<T>
  and requires(T t)
  {
    typename std::tuple_size<T>::type;

    requires std::derived_from<
      std::tuple_size<T>,
      std::integral_constant<std::size_t, std::tuple_size_v<T>>
    >;

  }
  and has_tuple_elements<T>(std::make_index_sequence<std::tuple_size_v<T>>{})
;


template<class T>
concept tuple_like = tuple_like_impl<std::remove_cvref_t<T>>;


static_assert(tuple_like<std::tuple<>>);
static_assert(tuple_like<std::pair<float,double>>);
static_assert(tuple_like<std::tuple<int>>);
static_assert(tuple_like<std::tuple<int,int,int>>);
static_assert(tuple_like<std::tuple<int,int,int,float>>);
static_assert(tuple_like<std::array<int,10>>);
static_assert(not tuple_like<int>);
static_assert(not tuple_like<float>);


template<class T, std::size_t N>
concept tuple_like_of_size =
  tuple_like<T>
  and (std::tuple_size_v<std::remove_cvref_t<T>> == N)
;


template<class T>
concept unit_like = tuple_like_of_size<T,0>;


template<class T>
concept pair_like = tuple_like_of_size<T,2>;


template<tuple_like T>
constexpr auto tuple_indices = std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>{};


// XXX tuple_of_indices would be unnecessary if std::index_sequence was itself a tuple-like
template<std::size_t... I>
constexpr auto tuple_of_indices_impl(std::index_sequence<I...>)
{
  return std::make_tuple(I...);
}

template<tuple_like T>
constexpr std::tuple tuple_of_indices = tuple_of_indices_impl(tuple_indices<T>);


template<std::size_t... I>
constexpr auto reversed_tuple_indices_impl(std::index_sequence<I...>)
{
  return std::index_sequence<(sizeof...(I) - 1 - I)...>{};
}

template<tuple_like T>
constexpr auto reversed_tuple_indices = reversed_tuple_indices_impl(tuple_indices<T>);


template<class T1, class... Ts>
concept same_tuple_size = 
  tuple_like<T1>
  and (... and tuple_like<Ts>)
  and (... and (std::tuple_size_v<std::remove_cvref_t<T1>> == std::tuple_size_v<std::remove_cvref_t<Ts>>))
;

static_assert(same_tuple_size<std::tuple<int>, std::tuple<int>>);
static_assert(same_tuple_size<std::tuple<int>, std::tuple<float>, std::array<double,1>>);
static_assert(same_tuple_size<std::pair<int,float>, std::array<double,2>, std::tuple<float,int>>);
static_assert(!same_tuple_size<std::pair<int,int>, std::tuple<float>, std::array<double,1>>);
static_assert(!same_tuple_size<std::tuple<float>, std::array<double,1>, std::tuple<>>);


template<class To, class From1, class... From>
concept all_convertible_to =
  std::convertible_to<From1,To>
  and (... and std::convertible_to<From,To>)
;


template<tuple_like T, class U, std::size_t... I>
  requires (std::tuple_size_v<std::remove_cvref_t<T>> > 0)
constexpr bool tuple_elements_convertible_to_impl(std::index_sequence<0,I...>)
{
  return all_convertible_to<U, std::tuple_element_t<0,T>, std::tuple_element_t<I,T>...>;
}


template<class FromTuple, class To>
concept tuple_elements_convertible_to =
  tuple_like<FromTuple>
  and (std::tuple_size_v<std::remove_cvref_t<FromTuple>> > 0)
  and tuple_elements_convertible_to_impl<FromTuple,To>(tuple_indices<FromTuple>)
;


template<std::size_t I, class T>
concept tuple_index_for =
  tuple_like<T>
  and (I < std::tuple_size_v<std::remove_cvref_t<T>>)
;


template<class T, std::size_t... I>
concept tuple_indices_for =
  tuple_like<T>
  and (... and tuple_index_for<I,T>)
;


template<class T>
struct is_std_array : std::false_type {};

template<class T, std::size_t N>
struct is_std_array<std::array<T,N>> : std::true_type {};


template<template<class...> class Tuple, class... Args>
constexpr Tuple<Args...> make_tuple_like(const Args&... args)
{
  using result_type = Tuple<Args...>;

  if constexpr(is_std_array<result_type>::value)
  {
    // std::array requires the weird doubly-nested brace syntax
    return result_type{{args...}};
  }
  else
  {
    return result_type{args...};
  }
}


// checks whether a list of types are all the same type
template<class Type, class... Types>
constexpr bool are_all_same()
{
  return (... and std::same_as<Type,Types>);
}

template<class... Types>
  requires (sizeof...(Types) < 2)
constexpr bool are_all_same()
{
  return true;
}

template<class... Types>
concept all_same = are_all_same<Types...>();


// checks whether some template can be instantiated with some list of types
template<template<class...> class Template, class... Types>
concept instantiatable = requires
{
  typename Template<Types...>;
};


// checks whether some type is an instance of some template instantiated with some list of types
// and those types can be rebound with some other list of new types
template<class T, class... Types>
struct is_rebindable_with : std::false_type {};

template<template<class...> class Template, class... OldTypes, class... NewTypes>
struct is_rebindable_with<Template<OldTypes...>, NewTypes...> : std::integral_constant<bool, instantiatable<Template,NewTypes...>> {};


template<class T, class... NewTypes>
concept rebindable_with = is_rebindable_with<T,NewTypes...>::value;


// checks whether some type is an instance of some template similar to std::array
template<class T>
struct instantiated_like_std_array_impl : std::false_type {};

template<template <class,std::size_t> class Array, class T, std::size_t N>
struct instantiated_like_std_array_impl<Array<T,N>> : std::true_type {};

template<class T>
concept instantiated_like_std_array = instantiated_like_std_array_impl<T>::value;


// checks whether some type is an instance of some template similar to std::array
// and can be rebound like std::array using the list of Types...
template<class T, class... Types>
concept rebindable_like_std_array =
  instantiated_like_std_array<T>
  and all_same<Types...>
;


// takes a type that was instantiated similar to std::array<T,N> and rebinds T and N to different values
template<class A, class NewT, std::size_t NewN>
struct rebind_std_array_like;

template<template<class, std::size_t> class ArrayLike, class OldType, std::size_t OldN, class NewType, std::size_t NewN>
struct rebind_std_array_like<ArrayLike<OldType,OldN>, NewType, NewN>
{
  using type = ArrayLike<NewType,NewN>;
};


// takes a type that is an instance of some template with some list of types and rebinds that template with
// some other list of types
template<class T, class... NewTypes>
struct rebind_template;

template<template<class...> class Template, class... OldTypes, class... NewTypes>
struct rebind_template<Template<OldTypes...>, NewTypes...>
{
  using type = Template<NewTypes...>;
};


// takes a tuple_like types and rebinds the types of its elements
// the result is a new tuple_like type
template<tuple_like T, class... NewTypes>
struct rebind_tuple_like;


template<tuple_like T, class... Types, std::size_t... I>
constexpr bool are_tuple_elements_same_as(std::index_sequence<I...>)
{
  return (... and std::same_as<std::tuple_element_t<I, std::remove_cvref_t<T>>, Types>);
}


template<class T, class... Types>
concept tuple_elements_same_as =
  tuple_like<T>
  and (std::tuple_size_v<std::remove_cvref_t<T>> == sizeof...(Types))
  and are_tuple_elements_same_as<T,Types...>(tuple_indices<T>)
;


// case 0: the tuple_like's element types are the same as the types in NewTypes...
// there's no need to rebind to a new type
template<tuple_like T, class... NewTypes>
  requires tuple_elements_same_as<T, NewTypes...>
struct rebind_tuple_like<T, NewTypes...>
{
  using type = std::remove_cvref_t<T>;
};


// case 1: the tuple_like can be rebound with NewTypes... directly
template<tuple_like T, class... NewTypes>
  requires (not tuple_elements_same_as<T, NewTypes...>
            and rebindable_with<std::remove_cvref_t<T>, NewTypes...>)
struct rebind_tuple_like<T, NewTypes...>
{
  using type = typename rebind_template<std::remove_cvref_t<T>, NewTypes...>::type;
};


// case 2: the tuple_like can't be rebound with NewTypes... directly but the tuple_like can be rebound like a std::array
// rebind T like a std::array
template<tuple_like T, class... NewTypes>
  requires (not tuple_elements_same_as<T, NewTypes...>
            and not rebindable_with<std::remove_cvref_t<T>, NewTypes...>
            and rebindable_like_std_array<std::remove_cvref_t<T>, NewTypes...>)
struct rebind_tuple_like<T,NewTypes...>
{
  using old_array_type = std::remove_cvref_t<T>;

  // since all the types in NewTypes... are the same, the value type of the new array_like is the first type in this list
  // however, it's possible that the list NewTypes... is empty
  // in this case, just use the old value_type as the value type of the new array
  using new_value_type = std::tuple_element_t<0, std::tuple<NewTypes..., typename old_array_type::value_type>>;

  using type = typename rebind_std_array_like<old_array_type, new_value_type, sizeof...(NewTypes)>::type;
};


// case 3: the tuple_like cannot be rebound with NewTypes... directly and is not std::array-like
// and the number of NewTypes is 2, use a std::pair
template<tuple_like T, class... NewTypes>
  requires (not tuple_elements_same_as<T, NewTypes...>
            and not rebindable_with<std::remove_cvref_t<T>, NewTypes...>
            and not rebindable_like_std_array<std::remove_cvref_t<T>, NewTypes...>
            and sizeof...(NewTypes) == 2)
struct rebind_tuple_like<T,NewTypes...>
{
  using type = std::pair<NewTypes...>;
};


// case 4: the tuple_like cannot be rebound with NewTypes... directly and is not std::array-like
// the number of NewTypes is not 2, use a std::tuple
template<tuple_like T, class... NewTypes>
  requires (not tuple_elements_same_as<T, NewTypes...>
            and not rebindable_with<std::remove_cvref_t<T>, NewTypes...>
            and not rebindable_like_std_array<std::remove_cvref_t<T>, NewTypes...>
            and sizeof...(NewTypes) != 2)
struct rebind_tuple_like<T,NewTypes...>
{
  using type = std::tuple<NewTypes...>;
};


template<tuple_like Example, class... Types>
using smart_tuple = typename rebind_tuple_like<Example,Types...>::type;


template<class F, class I>
constexpr I fold_args(F, I init)
{
  return init;
}


template<class F, class I, class Arg, class... Args>
constexpr auto fold_args(F f, I init, Arg arg, Args... args)
{
  return fold_args(f, f(fold_args(f, init), arg), args...);
}


template<class F, class Arg, class... Args>
concept foldable_with =
  requires(F f, Arg arg1, Args... args)
  {
    fold_args(f, arg1, args...);
  }
;


template<class F, class I>
constexpr I fold_args_right(F, I init)
{
  return init;
}


template<class F, class I, class Arg, class... Args>
constexpr auto fold_args_right(F f, I init, Arg arg, Args... args)
{
  return fold_args_right(f, f(fold_args_right(f, init, args...), arg));
}


// tuple_fold
template<std::size_t... I, tuple_like T, class F>
constexpr decltype(auto) tuple_fold_impl(std::index_sequence<I...>, T&& t, F&& f)
{
  return fold_args(std::forward<F>(f), get<I>(std::forward<T>(t))...);
}


template<tuple_like T, class F>
constexpr auto tuple_fold(T&& t, F&& f)
{
  return tuple_fold_impl(tuple_indices<T>, std::forward<T>(t), std::forward<F>(f));
}


// tuple_fold_right
template<std::size_t... I, class Init, tuple_like T, class F>
constexpr decltype(auto) tuple_fold_right_impl(std::index_sequence<I...>, Init&& init, T&& t, F&& f)
{
  return fold_args_right(std::forward<F>(f), std::forward<Init>(init), get<I>(std::forward<T>(t))...);
}


template<class I, tuple_like T, class F>
constexpr auto tuple_fold_right(I&& init, T&& t, F&& f)
{
  return tuple_fold_right_impl(tuple_indices<T>, std::forward<I>(init), std::forward<T>(t), std::forward<F>(f));
}


template<class F, class T>
concept tuple_folder =
  tuple_like<T>
  and requires(T t, F f)
  {
    tuple_fold(t,f);
  }
;


// tuple_fold with init parameter
template<std::size_t... Is, class I, tuple_like T, class F>
constexpr auto tuple_fold_impl(std::index_sequence<Is...>, I&& init, T&& t, F&& f)
{
  return fold_args(std::forward<F>(f), std::forward<I>(init), get<Is>(std::forward<T>(t))...);
}

template<class I, tuple_like T, class F>
constexpr auto tuple_fold(I&& init, T&& t, F&& f)
{
  return tuple_fold_impl(tuple_indices<T>, std::forward<I>(init), std::forward<T>(t), std::forward<F>(f));
}


template<class F, std::size_t I, class... Tuples>
concept invocable_on_element =
  (... and tuple_like<Tuples>)
  and (... and tuple_index_for<I,Tuples>)
  and requires(F f, Tuples... tuples)
  {
    f(get<I>(tuples)...);
  }
;


template<class F, tuple_like... Tuples, std::size_t... I>
  requires (... and tuple_indices_for<Tuples, I...>)
constexpr bool is_invocable_elementwise_impl(std::index_sequence<I...>)
{
  return (... and invocable_on_element<F,I,Tuples...>);
}


template<class F>
constexpr bool is_invocable_elementwise()
{
  return std::is_invocable_v<F>;
}


template<class F, tuple_like Tuple1, tuple_like... Tuples>
  requires same_tuple_size<Tuple1,Tuples...>
constexpr bool is_invocable_elementwise()
{
  return is_invocable_elementwise_impl<F,Tuple1,Tuples...>(tuple_indices<Tuple1>);
}


template<class F, class... Tuples>
concept invocable_elementwise = is_invocable_elementwise<F,Tuples...>();


template<class R, class F, std::size_t I, class... Tuples>
concept invocable_r_on_element =
  invocable_on_element<F,I,Tuples...>
  and requires(F f, Tuples... tuples)
  {
    { f(get<I>(tuples)...) } -> std::convertible_to<R>;
  }
;


template<class R, class F, tuple_like... Tuples, std::size_t... I>
  requires (... and tuple_indices_for<Tuples, I...>)
constexpr bool is_invocable_r_elementwise_impl(std::index_sequence<I...>)
{
  return (... and invocable_r_on_element<R,F,I,Tuples...>);
}


template<class R, class F>
constexpr bool is_invocable_r_elementwise()
{
  return std::is_invocable_r_v<R,F>;
}


template<class R, class F, tuple_like Tuple1, tuple_like... Tuples>
  requires same_tuple_size<Tuple1,Tuples...>
constexpr bool is_invocable_r_elementwise()
{
  return is_invocable_r_elementwise_impl<R,F,Tuple1,Tuples...>(tuple_indices<Tuple1>);
}


template<class R, class F, class... Tuples>
concept invocable_r_elementwise = is_invocable_r_elementwise<R,F,Tuples...>();


template<std::size_t I, class F, tuple_like... Ts>
  requires (... and tuple_indices_for<Ts,I>)
constexpr decltype(auto) get_and_invoke(F&& f, Ts&&... ts)
{
  return std::invoke(std::forward<F>(f), get<I>(std::forward<Ts>(ts))...);
}

template<std::size_t I, class F, tuple_like... Ts>
  requires (... and tuple_indices_for<Ts,I>)
using get_and_invoke_result_t = decltype(get_and_invoke<I>(std::declval<F>(), std::declval<Ts>()...));


template<class F, class T, class... Ts>
concept tuple_zipper =
  tuple_like<T>
  and (... and tuple_like<Ts>)
  and same_tuple_size<T,Ts...>
  and invocable_elementwise<F,T,Ts...>
;


template<template<class...> class R, class F, tuple_like T, tuple_like... Ts, std::size_t... I>
  requires (tuple_zipper<F,T,Ts...>
            and sizeof...(I) == std::tuple_size_v<std::remove_cvref_t<T>>)
constexpr auto tuple_zip_with_r_impl(std::index_sequence<I...>, F&& f, T&& t, Ts&&... ts)
{
  return make_tuple_like<R>(get_and_invoke<I>(std::forward<F>(f), std::forward<T>(t), std::forward<Ts>(ts)...)...);
}


// this function zips the tuples by applying function f, and then passes the results of f as arguments to make_tuple_like<R>(...) and returns the resulting tuple_like
template<template<class...> class R, class F, tuple_like T, tuple_like... Ts, std::size_t... I>
  requires tuple_zipper<F,T,Ts...>
constexpr auto tuple_zip_with_r(F&& f, T&& t, Ts&&... ts)
{
  return tuple_zip_with_r_impl<R>(tuple_indices<T>, std::forward<F>(f), std::forward<T>(t), std::forward<Ts>(ts)...);
}


template<class R, class F, class T, class... Ts>
concept tuple_zipper_r = tuple_zipper<F,T,Ts...> and invocable_r_elementwise<R,F,T,Ts...>;


template<tuple_like T>
struct tuple_similar_to
{
  template<class... Types>
  using tuple = smart_tuple<T,Types...>;
};


// this makes a new tuple_like
// it attempts to make the type of the result similar to a preferred Example tuple_like
template<tuple_like Example, class... Args>
constexpr tuple_like auto make_tuple_similar_to(Args&&... args)
{
  return make_tuple_like<tuple_similar_to<Example>::template tuple>(std::forward<Args>(args)...);
}


// tuple_prepend_similar_to
template<tuple_like Example, std::size_t... I, tuple_like T, class Arg>
constexpr tuple_like auto tuple_prepend_similar_to_impl(std::index_sequence<I...>, T&& t, Arg&& arg)
{
  return make_tuple_similar_to<Example>(std::forward<Arg>(arg), get<I>(std::forward<T>(t))...);
}

template<tuple_like Example, tuple_like T, class Arg>
constexpr tuple_like auto tuple_prepend_similar_to(T&& t, Arg&& arg)
{
  return tuple_prepend_similar_to_impl<Example>(tuple_indices<T>, std::forward<T>(t), std::forward<Arg>(arg));
}


// tuple_prepend
template<tuple_like T, class Arg>
constexpr tuple_like auto tuple_prepend(const T& t, Arg&& arg)
{
  return tuple_prepend_similar_to<T>(t, std::forward<Arg>(arg));
}

// tuple_append_similar_to
template<tuple_like Example, std::size_t... I, tuple_like T, class Arg>
constexpr tuple_like auto tuple_append_similar_to_impl(std::index_sequence<I...>, T&& t, Arg&& arg)
{
  return make_tuple_similar_to<Example>(get<I>(std::forward<T>(t))..., std::forward<Arg>(arg));
}

template<tuple_like Example, tuple_like T, class Arg>
constexpr tuple_like auto tuple_append_similar_to(T&& t, Arg&& arg)
{
  return tuple_append_similar_to_impl<Example>(tuple_indices<T>, std::forward<T>(t), std::forward<Arg>(arg));
}

// tuple_append
template<tuple_like T, class Arg>
constexpr tuple_like auto tuple_append(const T& t, Arg&& arg)
{
  return tuple_append_similar_to<T>(t, std::forward<Arg>(arg));
}


template<tuple_like T, std::size_t... I>
  requires (std::tuple_size_v<T> == sizeof...(I))
constexpr tuple_like auto tuple_reverse_impl(std::index_sequence<I...>, const T& t)
{
  return make_tuple_similar_to<T>(get<I>(t)...);
}

template<tuple_like T>
constexpr tuple_like auto tuple_reverse(const T& t)
{
  return tuple_reverse_impl(reversed_tuple_indices<T>, t);
}


// tuple_zip_with zips the tuples by applying function f, and then returns the results of f as a smart_tuple

// 1-argument tuple_zip_with
template<tuple_like T, class F>
  requires tuple_zipper<F,T>
constexpr tuple_like auto tuple_zip_with(T&& t, F&& f)
{
  return tuple_zip_with_r<tuple_similar_to<T&&>::template tuple>(std::forward<F>(f), std::forward<T>(t));
}

// 2-argument tuple_zip_with
template<tuple_like T1, tuple_like T2, class F>
  requires tuple_zipper<F,T1,T2>
constexpr tuple_like auto tuple_zip_with(T1&& t1, T2&& t2, F&& f)
{
  return tuple_zip_with_r<tuple_similar_to<T1&&>::template tuple>(std::forward<F>(f), std::forward<T1>(t1), std::forward<T2>(t2));
}

// 3-argument tuple_zip_with
template<tuple_like T1, tuple_like T2, tuple_like T3, class F>
  requires tuple_zipper<F,T1,T2,T3>
constexpr tuple_like auto tuple_zip_with(T1&& t1, T2&& t2, T3&& t3, F&& f)
{
  return tuple_zip_with_r<tuple_similar_to<T1&&>::template tuple>(std::forward<F>(f), std::forward<T1>(t1), std::forward<T2>(t2), std::forward<T3>(t3));
}

// 4-argument tuple_zip_with
template<tuple_like T1, tuple_like T2, tuple_like T3, tuple_like T4, class F>
  requires tuple_zipper<F,T1,T2,T3,T4>
constexpr tuple_like auto tuple_zip_with(T1&& t1, T2&& t2, T3&& t3, T4&& t4, F&& f)
{
  return tuple_zip_with_r<tuple_similar_to<T1&&>::template tuple>(std::forward<F>(f), std::forward<T1>(t1), std::forward<T2>(t2), std::forward<T3>(t3), std::forward<T4>(t4));
}


template<class F, tuple_like T, tuple_like... Ts>
  requires (same_tuple_size<T,Ts...> and invocable_elementwise<F&&,T&&,Ts&&...>)
using tuple_zip_with_result_t = decltype(tuple_zip_with(std::declval<T>(), std::declval<Ts>()..., std::declval<F>()));


template<tuple_like T, tuple_like... Ts>
constexpr tuple_like auto tuple_zip(const T& t, const Ts&... ts)
{
  return tuple_zip_with(t, ts..., [](const auto&... elements)
  {
    return make_tuple_similar_to<T>(elements...);
  });
}


template<tuple_like T1, tuple_like T2, tuple_zipper<T1,T2> Op1, tuple_folder<tuple_zip_with_result_t<Op1,T1,T2>> Op2>
constexpr auto tuple_inner_product(const T1& t1, const T2& t2, Op1 star, Op2 plus)
{
  return tuple_fold(tuple_zip_with(t1, t2, star), plus);
}


template<tuple_like T1, tuple_like T2, tuple_like T3, tuple_zipper<T1,T2,T3> Op1, tuple_folder<tuple_zip_with_result_t<Op1,T1,T2,T3>> Op2>
constexpr auto tuple_transform_reduce(const T1& t1, const T2& t2, const T3& t3, Op1 star, Op2 plus)
{
  return tuple_fold(tuple_zip_with(t1, t2, t3, star), plus);
}


template<tuple_like T>
constexpr auto tuple_sum(const T& t)
{
  return tuple_fold(t, std::plus{});
}


template<tuple_like T>
constexpr auto tuple_product(const T& t)
{
  return tuple_fold(t, std::multiplies{});
}


template<tuple_like T1, tuple_like T2, tuple_zipper<T1,T2> Op>
  requires tuple_zipper_r<bool,Op,T1,T2>
constexpr bool tuple_equal(const T1& t1, const T2& t2, Op eq)
{
  return tuple_inner_product(t1, t2, eq, std::logical_and{});
}


template<tuple_like T1, tuple_like T2>
constexpr decltype(auto) tuple_equal(const T1& t1, const T2& t2)
{
  return tuple_equal(t1, t2, [](const auto& lhs, const auto& rhs)
  {
    return lhs == rhs;
  });
}


template<tuple_like T, class P>
  requires tuple_zipper_r<bool,P,T>
constexpr bool tuple_all(const T& t, const P& pred)
{
  auto folder = [&](bool partial_result, const auto& element)
  {
    return partial_result and pred(element);
  };

  return tuple_fold(t, folder);
}


template<class... Args>
void discard_args(Args&&...) {}


template<tuple_like T1, tuple_like T2, tuple_zipper<T1,T2> F, std::size_t... I>
  requires (sizeof...(I) == std::tuple_size_v<T1>)
constexpr void tuple_inplace_transform_impl(F&& f, T1& t1, const T2& t2, std::index_sequence<I...>)
{
  discard_args(get<I>(t1) = std::forward<F>(f)(get<I>(t1), get<I>(t2))...);
}


template<tuple_like T1, tuple_like T2, tuple_zipper<T1,T2> F>
constexpr void tuple_inplace_transform(F&& f, T1& t1, const T2& t2)
{
  return tuple_inplace_transform_impl(std::forward<F>(f), t1, t2, tuple_indices<T1>);
}


template<tuple_like T1, tuple_like T2, class C>
  requires same_tuple_size<T1,T2>
constexpr bool tuple_lexicographical_compare_impl(std::integral_constant<std::size_t, std::tuple_size_v<T1>>, const T1&, const T2&, const C&)
{
  return false;
}


template<std::size_t cursor, tuple_like T1, tuple_like T2, class C>
  requires (same_tuple_size<T1,T2> and cursor != std::tuple_size_v<T1>)
constexpr bool tuple_lexicographical_compare_impl(std::integral_constant<std::size_t, cursor>, const T1& t1, const T2& t2, const C& compare)
{
  if(compare(get<cursor>(t1), get<cursor>(t2))) return true;
  
  if(compare(get<cursor>(t2), get<cursor>(t1))) return false;
  
  return tuple_lexicographical_compare_impl(std::integral_constant<std::size_t,cursor+1>{}, t1, t2, compare);
}


template<tuple_like T1, tuple_like T2, class C>
  requires same_tuple_size<T1,T2>
constexpr bool tuple_lexicographical_compare(const T1& t1, const T2& t2, const C& compare)
{
  return tuple_lexicographical_compare_impl(std::integral_constant<std::size_t,0>{}, t1, t2, compare);
}


template<tuple_like T1, tuple_like T2>
  requires same_tuple_size<T1,T2>
constexpr bool tuple_lexicographical_compare(const T1& t1, const T2& t2)
{
  return tuple_lexicographical_compare(t1, t2, std::less{});
}


template<tuple_like T1, tuple_like T2, class C>
  requires same_tuple_size<T1,T2>
constexpr bool tuple_colexicographical_compare(const T1& t1, const T2& t2, const C& compare)
{
  return tuple_lexicographical_compare(tuple_reverse(t1), tuple_reverse(t2), compare);
}


template<tuple_like T1, tuple_like T2>
  requires same_tuple_size<T1,T2>
constexpr bool tuple_colexicographical_compare(const T1& t1, const T2& t2)
{
  return tuple_colexicographical_compare(t1, t2, std::less{});
}


template<tuple_like T, std::size_t Zero, std::size_t... I>
constexpr bool tuple_elements_have_same_tuple_size(std::index_sequence<Zero,I...>)
{
  using tuple_type = std::remove_cvref_t<T>;

  return same_tuple_size<
    std::tuple_element_t<0, tuple_type>,
    std::tuple_element_t<I, tuple_type>...
  >;
}


// treats the tuple t as a matrix and returns the Ith element of the Jth element of t
template<std::size_t I, std::size_t J, tuple_like T>
decltype(auto) get2d(T&& t)
{
  return get<I>(get<J>(std::forward<T>(t)));
}


template<std::size_t Row, std::size_t... Col, tuple_like T>
tuple_like auto tuple_unzip_row_impl(std::index_sequence<Col...>, T&& t)
{
  return make_tuple_like<tuple_similar_to<T>::template tuple>(get2d<Row,Col>(std::forward<T>(t))...);
}


template<std::size_t Row, tuple_like T>
tuple_like auto tuple_unzip_row(T&& t)
{
  return tuple_unzip_row_impl<Row>(tuple_indices<T>, std::forward<T>(t));
}


template<std::size_t... Row, tuple_like T>
tuple_like auto tuple_unzip_impl(std::index_sequence<Row...>, T&& t)
{
  using inner_tuple_type = std::tuple_element_t<0,std::remove_cvref_t<T>>;
  
  return make_tuple_like<tuple_similar_to<inner_tuple_type>::template tuple>
  (
    tuple_unzip_row<Row>(std::forward<T>(t))...
  );
}


// an unzippable_tuple_like it a tuple whose elements are each a tuple of size K
template<class T>
concept unzippable_tuple_like =
  tuple_like<T>
  and tuple_elements_have_same_tuple_size<T>(tuple_indices<T>);
;


// tuple_unzip takes a M-size tuple of N-size tuples (i.e., a matrix) and returns an N-size tuple of M-size tuples
// in other words, it returns the transpose of the matrix
//
// this probably has an elegant solution with tuple_zip_with or tuple_fold, but I don't know what it is
template<tuple_like T>
  requires unzippable_tuple_like<T>
tuple_like auto tuple_unzip(T&& t)
{
  using inner_tuple_type = std::tuple_element_t<0,std::remove_cvref_t<T>>;

  return tuple_unzip_impl(tuple_indices<inner_tuple_type>, std::forward<T>(t));
}


// the following concrete example demonstrates how tuple_unzip works with a tuple of pairs

//struct a{};
//struct b{};
//struct c{};
//struct d{};
//struct e{};
//struct f{};

//pair<tuple<a,b,c>, tuple<d,e,f>> tuple_unzip(tuple<pair<a,d>, pair<b,e>, pair<c,f>> t)
//{
//  using outer_tuple_type = decltype(t);
//  using inner_tuple_type = std::tuple_element_t<0,outer_tuple_type>;
//
//  // the idea is that t is a matrix and unzip is a transpose
//
//  // below, the column index goes from [0, 3), which are the indices of the outer tuple type
//  // the row index goes from [0, 2), which are the indices of the inner tuple type
//
//  return make_tuple_similar_to<inner_tuple_type>
//  (
//    make_tuple_like<tuple_similar_to<outer_tuple_type>::template tuple>(get<0>(get<0>(t)), get<0>(get<1>(t)), get<0>(get<2>(t))),
//    make_tuple_like<tuple_similar_to<outer_tuple_type>::template tuple>(get<1>(get<0>(t)), get<1>(get<1>(t)), get<1>(get<2>(t)))
//  );
//}


template<tuple_like T>
constexpr decltype(auto) tuple_last(T&& t)
{
  constexpr int N = std::tuple_size_v<std::remove_cvref_t<T>>;
  return get<N-1>(std::forward<T>(t));
}


template<std::size_t... I, tuple_like T>
constexpr tuple_like auto tuple_drop_impl(std::index_sequence<I...>, T&& t)
{
  constexpr std::size_t num_dropped = std::tuple_size_v<std::remove_cvref_t<T>> - sizeof...(I);

  return make_tuple_similar_to<T>(get<I+num_dropped>(std::forward<T>(t))...);
}


template<std::size_t N, tuple_like T>
  requires (N <= std::tuple_size_v<std::remove_cvref_t<T>>)
constexpr tuple_like auto tuple_drop(T&& t)
{
  constexpr std::size_t num_kept = std::tuple_size_v<std::remove_cvref_t<T>> - N;
  auto indices = std::make_index_sequence<num_kept>();
  return tuple_drop_impl(indices, std::forward<T>(t));
}


template<tuple_like T>
  requires (std::tuple_size_v<std::remove_cvref_t<T>> > 0)
constexpr tuple_like auto tuple_drop_first(T&& t)
{
  return tuple_drop<1>(std::forward<T>(t));
}


template<std::size_t... I, tuple_like T>
  requires (sizeof...(I) <= std::tuple_size_v<std::remove_cvref_t<T>>)
constexpr tuple_like auto tuple_take_impl(std::index_sequence<I...>, T&& t)
{
  return make_tuple_similar_to<T>(get<I>(t)...);
}


template<std::size_t N, tuple_like T>
  requires (N <= std::tuple_size_v<std::remove_cvref_t<T>>)
constexpr tuple_like auto tuple_take(T&& t)
{
  auto indices = std::make_index_sequence<N>();
  return tuple_take_impl(indices, std::forward<T>(t));
}


template<tuple_like T>
constexpr tuple_like auto tuple_drop_last(T&& t)
{
  constexpr int N = std::tuple_size_v<std::remove_cvref_t<T>>;
  return tuple_take<N-1>(std::forward<T>(t));
}


template<tuple_like R, std::size_t... I, tuple_like T>
constexpr tuple_like auto ensure_tuple_similar_to_impl(std::index_sequence<I...>, T&& t)
{
  return make_tuple_similar_to<R>(get<I>(std::forward<T>(t))...);
}


template<tuple_like R, tuple_like T>
constexpr tuple_like auto ensure_tuple_similar_to(T&& t)
{
  return ensure_tuple_similar_to_impl<R>(tuple_indices<T>, std::forward<T>(t));
}

template<tuple_like R, class T>
  requires (not tuple_like<T>)
constexpr tuple_like auto ensure_tuple_similar_to(T&& arg)
{
  return make_tuple_similar_to<R>(std::forward<T>(arg));
}


template<class T>
constexpr tuple_like auto ensure_tuple(T&& t)
{
  return ensure_tuple_similar_to<std::tuple<>>(std::forward<T>(t));
}


template<tuple_like T>
  requires (std::tuple_size_v<std::remove_cvref_t<T>> == 1)
constexpr decltype(auto) tuple_unwrap_single(T&& t)
{
  return get<0>(std::forward<T>(t));
}

template<tuple_like T>
  requires (std::tuple_size_v<std::remove_cvref_t<T>> != 1)
constexpr T&& tuple_unwrap_single(T&& t)
{
  return std::forward<T>(t);
}


template<tuple_like T>
constexpr auto tuple_drop_last_and_unwrap_single(T&& t)
{
  return tuple_unwrap_single(tuple_drop_last(std::forward<T>(t)));
}

template<bool do_wrap, class T>
  requires (do_wrap == true)
constexpr tuple_like auto tuple_wrap_if(T&& arg)
{
  return std::make_tuple(std::forward<T>(arg));
}

template<bool do_wrap, class T>
  requires (do_wrap == false)
constexpr auto tuple_wrap_if(T&& arg)
{
  return std::forward<T>(arg);
}


template<tuple_like R, std::size_t... I, std::size_t... J, tuple_like T1, tuple_like T2>
constexpr tuple_like auto tuple_cat_similar_to_impl(std::index_sequence<I...>, std::index_sequence<J...>, T1&& t1, T2&& t2)
{
  return make_tuple_similar_to<R>(get<I>(std::forward<T1>(t1))..., get<J>(std::forward<T2>(t2))...);
}


template<tuple_like R>
constexpr tuple_like auto tuple_cat_similar_to()
{
  return make_tuple_similar_to<R>();
}


template<tuple_like R, tuple_like T, std::size_t... I>
constexpr tuple_like auto tuple_cat_similar_to_impl(std::index_sequence<I...>, T&& t)
{
  return make_tuple_similar_to<R>(get<I>(std::forward<T>(t))...);
}


template<tuple_like R, tuple_like T>
constexpr tuple_like auto tuple_cat_similar_to(T&& t)
{
  return tuple_cat_similar_to_impl<T>(tuple_indices<T>, std::forward<T>(t));
}


template<tuple_like R, tuple_like T1, tuple_like T2>
constexpr tuple_like auto tuple_cat_similar_to(T1&& t1, T2&& t2)
{
  return tuple_cat_similar_to_impl<R>(tuple_indices<T1>, tuple_indices<T2>, std::forward<T1>(t1), std::forward<T2>(t2));
}


template<tuple_like R, tuple_like T1, tuple_like T2, tuple_like... Ts>
constexpr tuple_like auto tuple_cat_similar_to(T1&& t1, T2&& t2, Ts&&... ts)
{
  return tuple_cat_similar_to<R>(tuple_cat_similar_to<R>(std::forward<T1>(t1), std::forward<T2>(t2)), std::forward<Ts>(ts)...);
}


template<tuple_like T, std::size_t... I>
constexpr tuple_like auto tuple_cat_all_impl(std::index_sequence<I...>, T&& tuple_of_tuples)
{
  return tuple_cat_similar_to<T>(get<I>(std::forward<T>(tuple_of_tuples))...);
}


template<tuple_like T>
constexpr tuple_like auto tuple_cat_all(T&& tuple_of_tuples)
{
  return tuple_cat_all_impl(tuple_indices<T>, std::forward<T>(tuple_of_tuples));
}


constexpr std::tuple<> tuple_cat()
{
  return std::tuple();
}


template<tuple_like T, tuple_like... Ts>
constexpr tuple_like auto tuple_cat(const T& tuple, const Ts&... tuples)
{
  return tuple_cat_similar_to<T>(tuple, tuples...);
}


template<class Arg>
constexpr void output_args(std::ostream& os, const char*, const Arg& arg)
{
  os << arg;
}

template<class Arg, class... Args>
constexpr void output_args(std::ostream& os, const char* delimiter, const Arg& arg1, const Args&... args)
{
  os << arg1 << delimiter;

  output_args(os, delimiter, args...);
}

template<tuple_like T, std::size_t... Indices>
  requires (sizeof...(Indices) == std::tuple_size_v<T>)
constexpr std::ostream& tuple_output_impl(std::ostream& os, const char* delimiter, const T& t, std::index_sequence<Indices...>)
{
  output_args(os, delimiter, get<Indices>(t)...);
  return os;
}


template<tuple_like T>
constexpr std::ostream& tuple_output(std::ostream& os, const char* begin_tuple, const char* end_tuple, const char* delimiter, const T& t)
{
  os << begin_tuple;
  tuple_output_impl(os, delimiter, t, tuple_indices<T>);
  os << end_tuple;

  return os;
}


template<tuple_like T>
constexpr std::ostream& tuple_output(std::ostream& os, const T& t)
{
  return tuple_output(os, "(", ")", ", ", t);
}


template<class T>
constexpr tuple_like auto as_flat_tuple(const T& arg)
{
  if constexpr (not tuple_like<T>)
  {
    return std::make_tuple(arg);
  }
  else
  {
    auto tuple_of_tuples = tuple_zip_with(arg, [](const auto& element)
    {
      return as_flat_tuple(element);
    });

    return tuple_cat_all(tuple_of_tuples);
  }
}


// this function computes an inclusive scan on an input tuple
// the result of this function is a pair of values (result_tuple, carry_out):
//   1. a tuple with the same size as the input, and
//   2. a carry out value
// the user's combination function f has control over the value of the carry after each combination
// f(input[i], carry_in) must return the pair (result[i], carry_out)
template<tuple_like T, class C, class F>
constexpr auto tuple_inclusive_scan_with_carry(const T& input, const C& carry_in, const F& f)
{
  using namespace std;

  return tuple_fold(pair(tuple(), carry_in), input, [&f](const auto& prev_state, const auto& input_i)
  {
    // unpack the result of the previous fold iteration
    auto [prev_result, prev_carry] = prev_state;
    
    // combine the carry from the previous iteration with the current input element
    auto [result_i, carry_out] = f(input_i, prev_carry);

    // return the result of this iteration and the carry for the next iteration
    return pair(tuple_append_similar_to<T>(prev_result, result_i), carry_out);
  });
}


template<class T>
concept tuple_of_pair_like =
  unzippable_tuple_like<T>
  and pair_like<std::tuple_element_t<0,T>>
;

template<class T>
concept pair_of_not_tuple_like =
  pair_like<T>
  and (not tuple_like<std::tuple_element_t<0,T>>)
;

template<pair_of_not_tuple_like T>
constexpr T unzip_innermost_pairs(const T& pair)
{
  return pair;
}

template<tuple_of_pair_like T>
constexpr pair_like auto unzip_innermost_pairs(const T& tuple_of_pairs)
{
  // this will return a pair of tuples
  return tuple_unzip(tuple_of_pairs);
}

template<tuple_like T>
  requires (not tuple_of_pair_like<T> and not pair_of_not_tuple_like<T>)
constexpr pair_like auto unzip_innermost_pairs(const T& tuple)
{
  // this will return a pair of tuples
  return tuple_unzip(tuple_zip_with(tuple, [](auto t_i)
  {
    return unzip_innermost_pairs(t_i);
  }));
}


template<ubu::detail::tuple_like T, class F, std::size_t... I>
constexpr auto unpack_and_invoke_impl(T&& arg, F&& f, std::index_sequence<I...>)
{
  return std::invoke(std::forward<F>(f), get<I>(std::forward<T>(arg))...);
}

template<ubu::detail::tuple_like T, class F>
constexpr auto unpack_and_invoke(T&& args, F&& f)
{
  auto indices = tuple_indices<T>;
  return unpack_and_invoke_impl(std::forward<T>(args), std::forward<F>(f), indices);
}


template<tuple_like R, tuple_like T, class F, std::size_t... I>
constexpr tuple_like auto tuple_static_enumerate_similar_to_impl(const T& tuple, F&& f, std::index_sequence<I...>)
{
  return make_tuple_similar_to<R>(f.template operator()<I>(get<I>(tuple))...); 
}

template<tuple_like R, tuple_like T, class F>
constexpr tuple_like auto tuple_static_enumerate_similar_to(const T& tuple, F&& f)
{
  return tuple_static_enumerate_similar_to_impl<R>(tuple, std::forward<F>(f), tuple_indices<T>);
}

template<tuple_like T, class F>
constexpr tuple_like auto tuple_static_enumerate(const T& tuple, F&& f)
{
  return tuple_static_enumerate_similar_to<T>(tuple, std::forward<F>(f));
}


template<std::size_t I, tuple_like T, class U>
  requires tuple_index_for<I,T>
constexpr tuple_like auto tuple_replace_element(const T& tuple, const U& replacement)
{
  return tuple_static_enumerate(tuple, [&]<std::size_t index>(const auto& t_i)
  {
    if constexpr (index == I)
    {
      return replacement;
    }
    else
    {
      return t_i;
    }
  });
}


} // end ubu::detail


#include "../../../detail/epilogue.hpp"

