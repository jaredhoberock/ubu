#pragma once

#include "../../detail/prologue.hpp"

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
    //{ get<N>(t) } -> std::convertible_to<const std::tuple_element_t<N, T>&>;
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


template<tuple_like T>
constexpr auto tuple_indices = std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>{};


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


template<template<class...> class Template, class... Types>
concept instantiatable = requires
{
  typename Template<Types...>;
};


template<class T, class... Types>
struct is_rebindable_with : std::false_type {};

template<template<class...> class Template, class... OldTypes, class... NewTypes>
struct is_rebindable_with<Template<OldTypes...>, NewTypes...> : std::integral_constant<bool, instantiatable<Template,NewTypes...>> {};


template<class T, class... NewTypes>
concept rebindable_with = is_rebindable_with<T,NewTypes...>::value;


template<class T, class... NewTypes>
concept tuple_like_rebindable_with = (tuple_like<T> and rebindable_with<T,NewTypes...>);


template<class MaybeTupleLike, class... NewTypes>
struct rebind_tuple_like;


// case 0: T is not tuple_like and there are two types to bind
// use a std::pair
template<class T, class... NewTypes>
  requires (not tuple_like<T> and sizeof...(NewTypes) == 2)
struct rebind_tuple_like<T,NewTypes...>
{
  using type = std::pair<NewTypes...>;
};


// case 1: T is not tuple_like and there are some number other than two types to bind
// use a std::tuple
template<class T, class... NewTypes>
  requires (not tuple_like<T> and sizeof...(NewTypes) != 2)
struct rebind_tuple_like<T,NewTypes...>
{
  using type = std::tuple<NewTypes...>;
};

// case 2: T is tuple_like and bindable like a tuple and there are heterogeneous types to bind
template<template<class...> class TupleLike, class... OldTypes, class... NewTypes>
  requires tuple_like_rebindable_with<TupleLike<OldTypes...>, NewTypes...>
struct rebind_tuple_like<TupleLike<OldTypes...>, NewTypes...>
{
  using type = TupleLike<NewTypes...>;
};

// case 3: T is tuple_like and bindable like an array and there are homogeneous types to bind
// (i.e. T is array_like)
template<template<class,std::size_t> class ArrayLike, class OldType, std::size_t N, class NewType, class... NewTypes>
  requires (tuple_like<ArrayLike<OldType,N>> and (std::same_as<NewType,NewTypes> and ...))
struct rebind_tuple_like<ArrayLike<OldType,N>, NewType, NewTypes...>
{
  using type = ArrayLike<NewType, 1 + sizeof...(NewTypes)>;
};


template<class Hint, class... Types>
using smart_tuple = typename rebind_tuple_like<std::remove_cvref_t<Hint>,Types...>::type;


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


template<std::size_t... I, tuple_like T, class Arg>
constexpr tuple_like auto tuple_append_impl(std::index_sequence<I...>, T&& t, Arg&& arg)
{
  return make_tuple_like<tuple_similar_to<T>::template tuple>(get<I>(std::forward<T>(t))..., std::forward<Arg>(arg));
}

template<tuple_like T, class Arg>
constexpr tuple_like auto tuple_append(const T& t, Arg&& arg)
{
  return tuple_append_impl(tuple_indices<T>, t, std::forward<Arg>(arg));
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


template<tuple_like T1, tuple_like T2, tuple_zipper<T1,T2> Op1, tuple_folder<tuple_zip_with_result_t<Op1,T1,T2>> Op2>
constexpr auto tuple_inner_product(const T1& t1, const T2& t2, Op1 star, Op2 plus)
{
  return tuple_fold(tuple_zip_with(t1, t2, star), plus);
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


template<tuple_like T, tuple_zipper_r<bool,T> P>
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


template<tuple_like T1, tuple_like T2>
  requires same_tuple_size<T1,T2>
constexpr bool tuple_lexicographical_compare_impl(std::integral_constant<std::size_t, std::tuple_size_v<T1>>, const T1& t1, const T2& t2)
{
  return false;
}


template<std::size_t cursor, tuple_like T1, tuple_like T2>
  requires (same_tuple_size<T1,T2> and cursor != std::tuple_size_v<T1>)
constexpr bool tuple_lexicographical_compare_impl(std::integral_constant<std::size_t, cursor>, const T1& t1, const T2& t2)
{
  if(get<cursor>(t1) < get<cursor>(t2)) return true;
  
  if(get<cursor>(t2) < get<cursor>(t1)) return false;
  
  return tuple_lexicographical_compare_impl(std::integral_constant<std::size_t,cursor+1>{}, t1, t2);
}


template<tuple_like T1, tuple_like T2>
  requires same_tuple_size<T1,T2>
constexpr bool tuple_lexicographical_compare(const T1& t1, const T2& t2)
{
  return tuple_lexicographical_compare_impl(std::integral_constant<std::size_t,0>{}, t1, t2);
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
//  return make_tuple_like<tuple_similar_to<inner_tuple_type>::template tuple>
//  (
//    make_tuple_like<tuple_similar_to<outer_tuple_type>::template tuple>(get<0>(get<0>(t)), get<0>(get<1>(t)), get<0>(get<2>(t))),
//    make_tuple_like<tuple_similar_to<outer_tuple_type>::template tuple>(get<1>(get<0>(t)), get<1>(get<1>(t)), get<1>(get<2>(t)))
//  );
//}


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


} // end ubu::detail


#include "../../detail/epilogue.hpp"

