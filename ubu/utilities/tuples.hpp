#pragma once

#include "../detail/prologue.hpp"

// aside from prologue.hpp, epilogue.hpp, and standard headers, this header shall remain standalone

#include <array>
#include <concepts>
#include <functional>
#include <iostream>
#include <tuple>
#include <utility>


// XXX TODO: in general, a lot of the functions inside namespace detail:: exist solely to get an index_sequence<I...>
//           such functions could be eliminated in favor of using std::apply tricks inside the function of interest

namespace ubu::tuples
{
namespace detail
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


} // end detail


template<class T>
concept tuple_like = detail::tuple_like_impl<std::remove_cvref_t<T>>;


static_assert(tuple_like<std::tuple<>>);
static_assert(tuple_like<std::pair<float,double>>);
static_assert(tuple_like<std::tuple<int>>);
static_assert(tuple_like<std::tuple<int,int,int>>);
static_assert(tuple_like<std::tuple<int,int,int,float>>);
static_assert(tuple_like<std::array<int,10>>);
static_assert(not tuple_like<int>);
static_assert(not tuple_like<float>);


template<tuple_like T>
constexpr auto size_v = std::tuple_size_v<std::remove_cvref_t<T>>;


template<class T, std::size_t N>
concept tuple_like_of_size =
  tuple_like<T>
  and (N == size_v<T>)
;

template<class T, std::size_t N>
concept tuple_like_of_size_at_least =
  tuple_like<T>
  and (N <= size_v<T>)
;


template<std::size_t I, tuple_like_of_size_at_least<I+1> T>
using element_t = std::tuple_element_t<I, std::remove_cvref_t<T>>;


template<class T>
concept unit_like = tuple_like_of_size<T,0>;

template<class T>
concept single_like = tuple_like_of_size<T,1>;

template<class T>
concept pair_like = tuple_like_of_size<T,2>;


template<tuple_like_of_size_at_least<1> T>
constexpr decltype(auto) first(T&& t)
{
  return get<0>(std::forward<T>(t));
}

template<tuple_like_of_size_at_least<1> T>
using first_t = element_t<0,T>;


template<tuple_like_of_size_at_least<2> T>
constexpr decltype(auto) second(T&& t)
{
  return get<1>(std::forward<T>(t));
}

template<tuple_like_of_size_at_least<2> T>
using second_t = element_t<1,T>;


template<tuple_like_of_size_at_least<3> T>
constexpr decltype(auto) third(T&& t)
{
  return get<2>(std::forward<T>(t));
}

template<tuple_like_of_size_at_least<3> T>
using third_t = element_t<2,T>;


template<tuple_like T>
constexpr auto indices_v = std::make_index_sequence<size_v<T>>{};


namespace detail
{


template<std::size_t... I>
constexpr auto tuple_of_indices_impl(std::index_sequence<I...>)
{
  return std::make_tuple(I...);
}

} // end detail


// XXX tuple_of_indices_v would be unnecessary if std::index_sequence was itself a tuple-like
template<tuple_like T>
constexpr std::tuple tuple_of_indices_v = detail::tuple_of_indices_impl(indices_v<T>);


namespace detail
{


template<std::size_t... I>
constexpr auto reversed_tuple_indices_impl(std::index_sequence<I...>)
{
  return std::index_sequence<(sizeof...(I) - 1 - I)...>{};
}


} // end detail


template<tuple_like T>
constexpr auto reversed_indices_v = detail::reversed_tuple_indices_impl(indices_v<T>);


template<class T1, class... Ts>
concept same_size = 
  tuple_like<T1>
  and (... and tuple_like<Ts>)
  and (... and (size_v<T1> == size_v<Ts>))
;

static_assert(same_size<std::tuple<int>, std::tuple<int>>);
static_assert(same_size<std::tuple<int>, std::tuple<float>, std::array<double,1>>);
static_assert(same_size<std::pair<int,float>, std::array<double,2>, std::tuple<float,int>>);
static_assert(!same_size<std::pair<int,int>, std::tuple<float>, std::array<double,1>>);
static_assert(!same_size<std::tuple<float>, std::array<double,1>, std::tuple<>>);


namespace detail
{


template<class To, class From1, class... From>
concept all_convertible_to =
  std::convertible_to<From1,To>
  and (... and std::convertible_to<From,To>)
;


template<tuple_like T, class U, std::size_t... I>
  requires (size_v<T> > 0)
constexpr bool all_elements_convertible_to_impl(std::index_sequence<0,I...>)
{
  return all_convertible_to<U, element_t<0,T>, element_t<I,T>...>;
}


} // end detail


template<class FromTuple, class To>
concept all_elements_convertible_to =
  tuple_like<FromTuple>
  and (size_v<FromTuple> > 0)
  and detail::all_elements_convertible_to_impl<FromTuple,To>(indices_v<FromTuple>)
;


template<std::size_t I, class T>
concept is_index_for =
  tuple_like<T>
  and (I < size_v<T>)
;


template<class T, std::size_t... I>
concept are_indices_for =
  tuple_like<T>
  and (... and is_index_for<I,T>)
;


namespace detail
{


template<class T>
struct is_std_array : std::false_type {};

template<class T, std::size_t N>
struct is_std_array<std::array<T,N>> : std::true_type {};


} // end detail


// Most of the following functions return a tuple. The type of the tuple returned depends on which function variant you call.
//
// When a function is suffixed with "_r", the first template parameter is a class template which will be used to instantiate the type of the result tuple.
//
// Example:
//
//    // decltype(result) is std::tuple<int, const char*, float>
//    auto result = make_r<std::tuple>(0, "foo", 13.f);
//
// In most other cases, the returned type of tuple is deduced to be "like" some example tuple_like type.
// Here, "like" means that the result tries to be another instantiation of the same template from which the example was instantiated. 
// If that's not possible, the result usually defaults to some instantiation of std::tuple.
// 
// When a function is suffixed with "_like", the example tuple_like type is the explicit, first template parameter.
// Unlike the "_r" variants, the first, explicit template parameter is a type, not a template.
//
// Example:
//  
//     std::tuple single(0);       // a std::tuple<int>
//     std::pair p(0, 13.f);       // a std::pair<int, float>
//
//     // decltype(result) is std::pair<int,float>, not std::tuple<int,float>
//     auto result = append_like<decltype(p)>(single, 42.f);
//
//
// When a function has no suffix, the example tuple_like type is the first parameter of the function.
//
// Example:
//
//     std::tuple single(0); // a std::tuple<int>
//
//     // decltype(result) is std::tuple<int,float>, not std::pair<int,float>
//     auto result = append(single, 42.f);
//
// Note that not all functions have all variants.



template<template<class...> class ResultTuple, class... Args>
constexpr ResultTuple<Args...> make_r(const Args&... args)
{
  using result_type = ResultTuple<Args...>;

  if constexpr(detail::is_std_array<result_type>::value)
  {
    // std::array requires the weird doubly-nested brace syntax
    return result_type{{args...}};
  }
  else
  {
    return result_type{args...};
  }
}


namespace detail
{


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
constexpr bool are_all_elements_same_as(std::index_sequence<I...>)
{
  return (... and std::same_as<element_t<I,T>, Types>);
}


template<class T, class... Types>
concept all_elements_same_as =
  tuple_like<T>
  and (size_v<T> == sizeof...(Types))
  and are_all_elements_same_as<T,Types...>(indices_v<T>)
;


// case 0: the tuple_like's element types are the same as the types in NewTypes...
// there's no need to rebind to a new type
template<tuple_like T, class... NewTypes>
  requires all_elements_same_as<T, NewTypes...>
struct rebind_tuple_like<T, NewTypes...>
{
  using type = std::remove_cvref_t<T>;
};


// case 1: the tuple_like can be rebound with NewTypes... directly
template<tuple_like T, class... NewTypes>
  requires (not all_elements_same_as<T, NewTypes...>
            and rebindable_with<std::remove_cvref_t<T>, NewTypes...>)
struct rebind_tuple_like<T, NewTypes...>
{
  using type = typename rebind_template<std::remove_cvref_t<T>, NewTypes...>::type;
};


// case 2: the tuple_like can't be rebound with NewTypes... directly but the tuple_like can be rebound like a std::array
// rebind T like a std::array
template<tuple_like T, class... NewTypes>
  requires (not all_elements_same_as<T, NewTypes...>
            and not rebindable_with<std::remove_cvref_t<T>, NewTypes...>
            and rebindable_like_std_array<std::remove_cvref_t<T>, NewTypes...>)
struct rebind_tuple_like<T,NewTypes...>
{
  using old_array_type = std::remove_cvref_t<T>;

  // since all the types in NewTypes... are the same, the value type of the new array_like is the first type in this list
  // however, it's possible that the list NewTypes... is empty
  // in this case, just use the old value_type as the value type of the new array
  using new_value_type = element_t<0, std::tuple<NewTypes..., typename old_array_type::value_type>>;

  using type = typename rebind_std_array_like<old_array_type, new_value_type, sizeof...(NewTypes)>::type;
};


// case 3: the tuple_like cannot be rebound with NewTypes... directly and is not std::array-like
// and the number of NewTypes is 2, use a std::pair
template<tuple_like T, class... NewTypes>
  requires (not all_elements_same_as<T, NewTypes...>
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
  requires (not all_elements_same_as<T, NewTypes...>
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
constexpr I fold_args_left(F, I init)
{
  return init;
}


template<class F, class I, class Arg, class... Args>
constexpr auto fold_args_left(F f, I init, Arg arg, Args... args)
{
  return detail::fold_args_left(f, f(detail::fold_args_left(f, init), arg), args...);
}


template<class F, class Arg, class... Args>
concept foldable_with =
  requires(F f, Arg arg1, Args... args)
  {
    detail::fold_args_left(f, arg1, args...);
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
  return detail::fold_args_right(f, f(detail::fold_args_right(f, init, args...), arg));
}


template<std::size_t... I, tuple_like T, class F>
constexpr decltype(auto) fold_left_impl(std::index_sequence<I...>, T&& t, F&& f)
{
  return detail::fold_args_left(std::forward<F>(f), get<I>(std::forward<T>(t))...);
}


} // end detail


// fold_left with no init parameter
template<tuple_like T, class F>
constexpr auto fold_left(T&& t, F&& f)
{
  return detail::fold_left_impl(indices_v<T>, std::forward<T>(t), std::forward<F>(f));
}


namespace detail
{


template<std::size_t... I, class Init, tuple_like T, class F>
constexpr decltype(auto) fold_right_impl(std::index_sequence<I...>, Init&& init, T&& t, F&& f)
{
  return detail::fold_args_right(std::forward<F>(f), std::forward<Init>(init), get<I>(std::forward<T>(t))...);
}


} // end detail


// fold_right with no init parameter
template<class I, tuple_like T, class F>
constexpr auto fold_right(I&& init, T&& t, F&& f)
{
  return detail::fold_right_impl(indices_v<T>, std::forward<I>(init), std::forward<T>(t), std::forward<F>(f));
}


template<class F, class T>
concept left_folder =
  tuple_like<T>
  and requires(T t, F f)
  {
    fold_left(t,f);
  }
;


namespace detail
{


template<std::size_t... Is, class I, tuple_like T, class F>
constexpr auto fold_left_impl(std::index_sequence<Is...>, I&& init, T&& t, F&& f)
{
  return detail::fold_args_left(std::forward<F>(f), std::forward<I>(init), get<Is>(std::forward<T>(t))...);
}


} // end detail


// fold_left with init parameter
template<class I, tuple_like T, class F>
constexpr auto fold_left(I&& init, T&& t, F&& f)
{
  return detail::fold_left_impl(indices_v<T>, std::forward<I>(init), std::forward<T>(t), std::forward<F>(f));
}


namespace detail
{


template<class F, std::size_t I, class... Tuples>
concept invocable_on_element =
  (... and tuple_like<Tuples>)
  and (... and is_index_for<I,Tuples>)
  and requires(F f, Tuples... tuples)
  {
    f(get<I>(tuples)...);
  }
;


template<class F, tuple_like... Tuples, std::size_t... I>
  requires (... and are_indices_for<Tuples, I...>)
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
  requires same_size<Tuple1,Tuples...>
constexpr bool is_invocable_elementwise()
{
  return is_invocable_elementwise_impl<F,Tuple1,Tuples...>(indices_v<Tuple1>);
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
  requires (... and are_indices_for<Tuples, I...>)
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
  requires same_size<Tuple1,Tuples...>
constexpr bool is_invocable_r_elementwise()
{
  return is_invocable_r_elementwise_impl<R,F,Tuple1,Tuples...>(indices_v<Tuple1>);
}


template<class R, class F, class... Tuples>
concept invocable_r_elementwise = is_invocable_r_elementwise<R,F,Tuples...>();


template<std::size_t I, class F, tuple_like... Ts>
  requires (... and are_indices_for<Ts,I>)
constexpr decltype(auto) get_and_invoke(F&& f, Ts&&... ts)
{
  return std::invoke(std::forward<F>(f), get<I>(std::forward<Ts>(ts))...);
}

template<std::size_t I, class F, tuple_like... Ts>
  requires (... and are_indices_for<Ts,I>)
using get_and_invoke_result_t = decltype(get_and_invoke<I>(std::declval<F>(), std::declval<Ts>()...));


} // end detail


template<class F, class T, class... Ts>
concept zipper =
  tuple_like<T>
  and (... and tuple_like<Ts>)
  and same_size<T,Ts...>
  and detail::invocable_elementwise<F,T,Ts...>
;


namespace detail
{


template<template<class...> class R, class F, tuple_like T, tuple_like... Ts, std::size_t... I>
  requires (zipper<F,T,Ts...>
            and sizeof...(I) == size_v<T>)
constexpr auto zip_with_r_impl(std::index_sequence<I...>, F&& f, T&& t, Ts&&... ts)
{
  return tuples::make_r<R>(detail::get_and_invoke<I>(std::forward<F>(f), std::forward<T>(t), std::forward<Ts>(ts)...)...);
}


} // end detail


// this function zips the tuples by applying function f, and then passes the results of f as arguments to make_r<R>(...) and returns the resulting tuple_like
template<template<class...> class R, class F, tuple_like T, tuple_like... Ts>
  requires zipper<F,T,Ts...>
constexpr auto zip_with_r(F&& f, T&& t, Ts&&... ts)
{
  return detail::zip_with_r_impl<R>(indices_v<T>, std::forward<F>(f), std::forward<T>(t), std::forward<Ts>(ts)...);
}


template<class R, class F, class T, class... Ts>
concept zipper_r = zipper<F,T,Ts...> and detail::invocable_r_elementwise<R,F,T,Ts...>;


namespace detail
{


template<tuple_like T>
struct tuple_template_like
{
  template<class... Types>
  using tuple = smart_tuple<T,Types...>;
};


} // end detail


// this makes a new tuple_like
// it attempts to make the type of the result similar to a preferred Example tuple_like
template<tuple_like Example, class... Args>
constexpr tuple_like auto make_like(Args&&... args)
{
  return tuples::make_r<detail::tuple_template_like<Example>::template tuple>(std::forward<Args>(args)...);
}


namespace detail
{


template<tuple_like Example, std::size_t... I, tuple_like T, class Arg>
constexpr tuple_like auto prepend_like_impl(std::index_sequence<I...>, T&& t, Arg&& arg)
{
  return tuples::make_like<Example>(std::forward<Arg>(arg), get<I>(std::forward<T>(t))...);
}


} // end detail


// prepend_like
template<tuple_like Example, tuple_like T, class Arg>
constexpr tuple_like auto prepend_like(T&& t, Arg&& arg)
{
  return detail::prepend_like_impl<Example>(indices_v<T>, std::forward<T>(t), std::forward<Arg>(arg));
}


// prepend
template<tuple_like T, class Arg>
constexpr tuple_like auto prepend(const T& t, Arg&& arg)
{
  return tuples::prepend_like<T>(t, std::forward<Arg>(arg));
}


namespace detail
{


template<tuple_like Example, std::size_t... I, tuple_like T, class Arg>
constexpr tuple_like auto append_like_impl(std::index_sequence<I...>, T&& t, Arg&& arg)
{
  return tuples::make_like<Example>(get<I>(std::forward<T>(t))..., std::forward<Arg>(arg));
}


} // end detail

// append_like
template<tuple_like Example, tuple_like T, class Arg>
constexpr tuple_like auto append_like(T&& t, Arg&& arg)
{
  return detail::append_like_impl<Example>(indices_v<T>, std::forward<T>(t), std::forward<Arg>(arg));
}

// append
template<tuple_like T, class Arg>
constexpr tuple_like auto append(const T& t, Arg&& arg)
{
  return tuples::append_like<T>(t, std::forward<Arg>(arg));
}


namespace detail
{


template<tuple_like T, std::size_t... I>
  requires (size_v<T> == sizeof...(I))
constexpr tuple_like auto reverse_impl(std::index_sequence<I...>, const T& t)
{
  return tuples::make_like<T>(get<I>(t)...);
}


} // end detail


// reverse
template<tuple_like T>
constexpr tuple_like auto reverse(const T& t)
{
  return detail::reverse_impl(reversed_indices_v<T>, t);
}


// zip_with zips the tuples by applying function f, and then returns the results of f as a smart_tuple

// 1-argument zip_with
template<tuple_like T, zipper<T> F>
constexpr tuple_like auto zip_with(T&& t, F&& f)
{
  return zip_with_r<detail::tuple_template_like<T&&>::template tuple>(std::forward<F>(f), std::forward<T>(t));
}

// 2-argument zip_with
template<tuple_like T1, tuple_like T2, zipper<T1,T2> F>
constexpr tuple_like auto zip_with(T1&& t1, T2&& t2, F&& f)
{
  return zip_with_r<detail::tuple_template_like<T1&&>::template tuple>(std::forward<F>(f), std::forward<T1>(t1), std::forward<T2>(t2));
}

// 3-argument zip_with
template<tuple_like T1, tuple_like T2, tuple_like T3, zipper<T1,T2,T3> F>
constexpr tuple_like auto zip_with(T1&& t1, T2&& t2, T3&& t3, F&& f)
{
  return zip_with_r<detail::tuple_template_like<T1&&>::template tuple>(std::forward<F>(f), std::forward<T1>(t1), std::forward<T2>(t2), std::forward<T3>(t3));
}

// 4-argument zip_with
template<tuple_like T1, tuple_like T2, tuple_like T3, tuple_like T4, zipper<T1,T2,T3,T4> F>
constexpr tuple_like auto zip_with(T1&& t1, T2&& t2, T3&& t3, T4&& t4, F&& f)
{
  return zip_with_r<detail::tuple_template_like<T1&&>::template tuple>(std::forward<F>(f), std::forward<T1>(t1), std::forward<T2>(t2), std::forward<T3>(t3), std::forward<T4>(t4));
}


template<class F, tuple_like T, tuple_like... Ts>
  requires (same_size<T,Ts...> and detail::invocable_elementwise<F&&,T&&,Ts&&...>)
using zip_with_result_t = decltype(zip_with(std::declval<T>(), std::declval<Ts>()..., std::declval<F>()));


template<tuple_like T, tuple_like... Ts>
constexpr tuple_like auto zip(const T& t, const Ts&... ts)
{
  return zip_with(t, ts..., [](const auto&... elements)
  {
    return tuples::make_like<T>(elements...);
  });
}


template<tuple_like T1, tuple_like T2, zipper<T1,T2> Op1, left_folder<zip_with_result_t<Op1,T1,T2>> Op2>
constexpr auto inner_product(const T1& t1, const T2& t2, Op1 star, Op2 plus)
{
  return tuples::fold_left(tuples::zip_with(t1, t2, star), plus);
}


template<tuple_like T1, tuple_like T2, tuple_like T3, zipper<T1,T2,T3> Op1, left_folder<zip_with_result_t<Op1,T1,T2,T3>> Op2>
constexpr auto tuple_transform_reduce(const T1& t1, const T2& t2, const T3& t3, Op1 star, Op2 plus)
{
  return tuples::fold_left(tuples::zip_with(t1, t2, t3, star), plus);
}


template<tuple_like T>
constexpr auto sum(const T& t)
{
  return tuples::fold_left(t, std::plus{});
}


template<tuple_like T>
constexpr auto product(const T& t)
{
  return tuples::fold_left(t, std::multiplies{});
}


template<tuple_like T1, tuple_like T2, zipper<T1,T2> Op>
  requires zipper_r<bool,Op,T1,T2>
constexpr bool equal(const T1& t1, const T2& t2, Op eq)
{
  return tuples::inner_product(t1, t2, eq, std::logical_and{});
}


template<tuple_like T1, tuple_like T2>
constexpr decltype(auto) equal(const T1& t1, const T2& t2)
{
  return tuples::equal(t1, t2, [](const auto& lhs, const auto& rhs)
  {
    return lhs == rhs;
  });
}


template<tuple_like T, class P>
  requires zipper_r<bool,P,T>
constexpr bool all_of(const T& t, const P& pred)
{
  auto folder = [&](bool partial_result, const auto& element)
  {
    return partial_result and pred(element);
  };

  return tuples::fold_left(true, t, folder);
}


namespace detail
{


template<class... Args>
void discard_args(Args&&...) {}


template<tuple_like T1, tuple_like T2, zipper<T1,T2> F, std::size_t... I>
  requires (sizeof...(I) == size_v<T1>)
constexpr void inplace_transform_impl(F&& f, T1& t1, const T2& t2, std::index_sequence<I...>)
{
  detail::discard_args(get<I>(t1) = std::forward<F>(f)(get<I>(t1), get<I>(t2))...);
}


} // end detail


template<tuple_like T1, tuple_like T2, zipper<T1,T2> F>
constexpr void inplace_transform(F&& f, T1& t1, const T2& t2)
{
  return detail::inplace_transform_impl(std::forward<F>(f), t1, t2, indices_v<T1>);
}


namespace detail
{


template<tuple_like T1, tuple_like T2, class C>
  requires same_size<T1,T2>
constexpr bool lexicographical_compare_impl(std::integral_constant<std::size_t, size_v<T1>>, const T1&, const T2&, const C&)
{
  return false;
}


template<std::size_t cursor, tuple_like T1, tuple_like T2, class C>
  requires (same_size<T1,T2> and cursor != size_v<T1>)
constexpr bool lexicographical_compare_impl(std::integral_constant<std::size_t, cursor>, const T1& t1, const T2& t2, const C& compare)
{
  if(compare(get<cursor>(t1), get<cursor>(t2))) return true;
  
  if(compare(get<cursor>(t2), get<cursor>(t1))) return false;
  
  return detail::lexicographical_compare_impl(std::integral_constant<std::size_t,cursor+1>{}, t1, t2, compare);
}


} // end detail


template<tuple_like T1, tuple_like T2, class C>
  requires same_size<T1,T2>
constexpr bool lexicographical_compare(const T1& t1, const T2& t2, const C& compare)
{
  return detail::lexicographical_compare_impl(std::integral_constant<std::size_t,0>{}, t1, t2, compare);
}


template<tuple_like T1, tuple_like T2>
  requires same_size<T1,T2>
constexpr bool lexicographical_compare(const T1& t1, const T2& t2)
{
  return tuples::lexicographical_compare(t1, t2, std::less{});
}


template<tuple_like T1, tuple_like T2, class C>
  requires same_size<T1,T2>
constexpr bool colexicographical_compare(const T1& t1, const T2& t2, const C& compare)
{
  return tuples::lexicographical_compare(tuples::reverse(t1), tuples::reverse(t2), compare);
}


template<tuple_like T1, tuple_like T2>
  requires same_size<T1,T2>
constexpr bool colexicographical_compare(const T1& t1, const T2& t2)
{
  return tuples::colexicographical_compare(t1, t2, std::less{});
}


namespace detail
{


template<tuple_like T, std::size_t Zero, std::size_t... I>
constexpr bool all_elements_have_same_size(std::index_sequence<Zero,I...>)
{
  using tuple_type = std::remove_cvref_t<T>;

  return same_size<
    element_t<0, tuple_type>,
    element_t<I, tuple_type>...
  >;
}


// treats the tuple t as a matrix and returns the Ith element of the Jth element of t
template<std::size_t I, std::size_t J, tuple_like T>
decltype(auto) get2d(T&& t)
{
  return get<I>(get<J>(std::forward<T>(t)));
}


template<std::size_t Row, std::size_t... Col, tuple_like T>
tuple_like auto unzip_row_impl(std::index_sequence<Col...>, T&& t)
{
  return tuples::make_r<tuple_template_like<T>::template tuple>(detail::get2d<Row,Col>(std::forward<T>(t))...);
}


template<std::size_t Row, tuple_like T>
tuple_like auto unzip_row(T&& t)
{
  return detail::unzip_row_impl<Row>(indices_v<T>, std::forward<T>(t));
}


template<std::size_t... Row, tuple_like T>
tuple_like auto unzip_impl(std::index_sequence<Row...>, T&& t)
{
  using inner_tuple_type = element_t<0,T>;
  
  return tuples::make_r<tuple_template_like<inner_tuple_type>::template tuple>
  (
    detail::unzip_row<Row>(std::forward<T>(t))...
  );
}


} // end detail


// an unzippable_tuple_like it a tuple whose elements are each a tuple of size K
template<class T>
concept unzippable_tuple_like =
  tuple_like<T>
  and detail::all_elements_have_same_size<T>(indices_v<T>);
;


// unzip takes a M-size tuple of N-size tuples (i.e., a matrix) and returns an N-size tuple of M-size tuples
// in other words, it returns the transpose of the matrix
//
// this probably has an elegant solution with zip_with or fold_left, but I don't know what it is
template<tuple_like T>
  requires unzippable_tuple_like<T>
tuple_like auto unzip(T&& t)
{
  using inner_tuple_type = element_t<0,T>;

  return detail::unzip_impl(indices_v<inner_tuple_type>, std::forward<T>(t));
}


// the following concrete example demonstrates how unzip works with a tuple of pairs

//struct a{};
//struct b{};
//struct c{};
//struct d{};
//struct e{};
//struct f{};

//pair<tuple<a,b,c>, tuple<d,e,f>> unzip(tuple<pair<a,d>, pair<b,e>, pair<c,f>> t)
//{
//  using outer_tuple_type = decltype(t);
//  using inner_tuple_type = element_t<0,outer_tuple_type>;
//
//  // the idea is that t is a matrix and unzip is a transpose
//
//  // below, the column index goes from [0, 3), which are the indices of the outer tuple type
//  // the row index goes from [0, 2), which are the indices of the inner tuple type
//
//  return make_like<inner_tuple_type>
//  (
//    make_r<tuple_template_like<outer_tuple_type>::template tuple>(get<0>(get<0>(t)), get<0>(get<1>(t)), get<0>(get<2>(t))),
//    make_r<tuple_template_like<outer_tuple_type>::template tuple>(get<1>(get<0>(t)), get<1>(get<1>(t)), get<1>(get<2>(t)))
//  );
//}

template<tuple_like T>
constexpr decltype(auto) last(T&& t)
{
  constexpr int N = size_v<T>;
  return get<N-1>(std::forward<T>(t));
}


namespace detail
{


template<tuple_like Example, std::size_t... I, tuple_like T>
constexpr tuple_like auto drop_like_impl(std::index_sequence<I...>, T&& t)
{
  constexpr std::size_t num_dropped = size_v<T> - sizeof...(I);

  return tuples::make_like<Example>(get<I+num_dropped>(std::forward<T>(t))...);
}


} // end detail


template<tuple_like Example, std::size_t N, tuple_like T>
  requires (N <= size_v<T>)
constexpr tuple_like auto drop_like(T&& t)
{
  constexpr std::size_t num_kept = size_v<T> - N;
  auto indices = std::make_index_sequence<num_kept>();
  return detail::drop_like_impl<Example>(indices, std::forward<T>(t));
}


template<std::size_t N, tuple_like T>
  requires (N <= size_v<T>)
constexpr tuple_like auto drop(T&& t)
{
  return tuples::drop_like<T,N>(std::forward<T>(t));
}


template<tuple_like T>
  requires (size_v<T> > 0)
constexpr tuple_like auto drop_first(T&& t)
{
  return tuples::drop<1>(std::forward<T>(t));
}


namespace detail
{


template<tuple_like Example, std::size_t... I, tuple_like T>
  requires (sizeof...(I) <= size_v<T>)
constexpr tuple_like auto take_like_impl(std::index_sequence<I...>, T&& t)
{
  return tuples::make_like<Example>(get<I>(t)...);
}


} // end detail


template<tuple_like Example, std::size_t N, tuple_like T>
  requires (N <= size_v<T>)
constexpr tuple_like auto take_like(T&& t)
{
  auto indices = std::make_index_sequence<N>();
  return detail::take_like_impl<Example>(indices, std::forward<T>(t));
}

template<std::size_t N, tuple_like T>
  requires (N <= size_v<T>)
constexpr tuple_like auto take(T&& t)
{
  return tuples::take_like<T,N>(std::forward<T>(t));
}

// this returns the leading elements of T, if there are any
// note that this will return a unit if T is single_like
template<tuple_like_of_size_at_least<2> T>
constexpr tuple_like auto leading(T&& t)
{
  return tuples::take<size_v<T>-1>(std::forward<T>(t));
}


template<tuple_like T>
constexpr tuple_like auto drop_last(T&& t)
{
  constexpr int N = size_v<T>;
  return tuples::take<N-1>(std::forward<T>(t));
}


namespace detail
{


template<tuple_like R, std::size_t... I, tuple_like T>
constexpr tuple_like auto ensure_tuple_like_impl(std::index_sequence<I...>, T&& t)
{
  return tuples::make_like<R>(get<I>(std::forward<T>(t))...);
}


} // end detail


template<tuple_like R, tuple_like T>
constexpr tuple_like auto ensure_tuple_like(T&& t)
{
  return detail::ensure_tuple_like_impl<R>(indices_v<T>, std::forward<T>(t));
}

template<tuple_like R, class T>
  requires (not tuple_like<T>)
constexpr tuple_like auto ensure_tuple_like(T&& arg)
{
  return tuples::make_like<R>(std::forward<T>(arg));
}


template<class T>
constexpr tuple_like auto ensure_tuple(T&& t)
{
  if constexpr(tuple_like<T>)
  {
    return std::forward<T>(t);
  }
  else
  {
    return tuples::ensure_tuple_like<std::tuple<>>(std::forward<T>(t));
  }
}


template<tuple_like T>
constexpr decltype(auto) unwrap_single(T&& t)
{
  if constexpr (single_like<T>)
  {
    return get<0>(std::forward<T>(t));
  }
  else
  {
    return std::forward<T>(t);
  }
}


template<tuple_like T>
constexpr auto drop_first_and_unwrap_single(T&& t)
{
  return tuples::unwrap_single(tuples::drop_first(std::forward<T>(t)));
}


template<tuple_like T>
constexpr auto drop_last_and_unwrap_single(T&& t)
{
  return tuples::unwrap_single(tuples::drop_last(std::forward<T>(t)));
}


template<bool do_wrap, class T>
  requires (do_wrap == true)
constexpr tuple_like auto wrap_if(T&& arg)
{
  return std::make_tuple(std::forward<T>(arg));
}

template<bool do_wrap, class T>
  requires (do_wrap == false)
constexpr auto wrap_if(T&& arg)
{
  return std::forward<T>(arg);
}


namespace detail
{


template<tuple_like R, std::size_t... I, std::size_t... J, tuple_like T1, tuple_like T2>
constexpr tuple_like auto concatenate_like_impl(std::index_sequence<I...>, std::index_sequence<J...>, T1&& t1, T2&& t2)
{
  return tuples::make_like<R>(get<I>(std::forward<T1>(t1))..., get<J>(std::forward<T2>(t2))...);
}


template<tuple_like R, tuple_like T, std::size_t... I>
constexpr tuple_like auto concatenate_like_impl(std::index_sequence<I...>, T&& t)
{
  return tuples::make_like<R>(get<I>(std::forward<T>(t))...);
}


} // end detail


template<tuple_like R>
constexpr tuple_like auto concatenate_like()
{
  return tuples::make_like<R>();
}


template<tuple_like R, tuple_like T>
constexpr tuple_like auto concatenate_like(T&& t)
{
  return detail::concatenate_like_impl<T>(indices_v<T>, std::forward<T>(t));
}


template<tuple_like R, tuple_like T1, tuple_like T2>
constexpr tuple_like auto concatenate_like(T1&& t1, T2&& t2)
{
  return detail::concatenate_like_impl<R>(indices_v<T1>, indices_v<T2>, std::forward<T1>(t1), std::forward<T2>(t2));
}


template<tuple_like R, tuple_like T1, tuple_like T2, tuple_like... Ts>
constexpr tuple_like auto concatenate_like(T1&& t1, T2&& t2, Ts&&... ts)
{
  return tuples::concatenate_like<R>(tuples::concatenate_like<R>(std::forward<T1>(t1), std::forward<T2>(t2)), std::forward<Ts>(ts)...);
}


namespace detail
{


template<tuple_like T, std::size_t... I>
constexpr tuple_like auto concatenate_all_impl(std::index_sequence<I...>, T&& tuple_of_tuples)
{
  return tuples::concatenate_like<T>(get<I>(std::forward<T>(tuple_of_tuples))...);
}


} // end detail


template<tuple_like T>
constexpr tuple_like auto concatenate_all(T&& tuple_of_tuples)
{
  return detail::concatenate_all_impl(indices_v<T>, std::forward<T>(tuple_of_tuples));
}


constexpr std::tuple<> concatenate()
{
  return std::tuple();
}


template<tuple_like T, tuple_like... Ts>
constexpr tuple_like auto concatenate(const T& tuple, const Ts&... tuples)
{
  return tuples::concatenate_like<T>(tuple, tuples...);
}


namespace detail
{


template<class Arg>
constexpr void output_args(std::ostream& os, const char*, const Arg& arg)
{
  os << arg;
}

template<class Arg, class... Args>
constexpr void output_args(std::ostream& os, const char* delimiter, const Arg& arg1, const Args&... args)
{
  os << arg1 << delimiter;

  detail::output_args(os, delimiter, args...);
}

template<tuple_like T, std::size_t... Indices>
  requires (sizeof...(Indices) == size_v<T>)
constexpr std::ostream& output_impl(std::ostream& os, const char* delimiter, const T& t, std::index_sequence<Indices...>)
{
  detail::output_args(os, delimiter, get<Indices>(t)...);
  return os;
}


} // end detail


template<tuple_like T>
constexpr std::ostream& output(std::ostream& os, const char* begin_tuple, const char* end_tuple, const char* delimiter, const T& t)
{
  os << begin_tuple;
  detail::output_impl(os, delimiter, t, indices_v<T>);
  os << end_tuple;

  return os;
}


template<tuple_like T>
constexpr std::ostream& output(std::ostream& os, const T& t)
{
  return tuples::output(os, "(", ")", ", ", t);
}


template<class T>
constexpr tuple_like auto flatten(const T& arg)
{
  if constexpr (not tuple_like<T>)
  {
    return std::make_tuple(arg);
  }
  else
  {
    auto tuple_of_tuples = tuples::zip_with(arg, [](const auto& element)
    {
      return tuples::flatten(element);
    });

    return tuples::concatenate_all(tuple_of_tuples);
  }
}


// this function computes an inclusive scan on an input tuple
// the result of this function is a pair of values (result_tuple, carry_out):
//   1. a tuple with the same size as the input, and
//   2. a carry out value (i.e., the result of fold_left)
// the user's combination function f has control over the value of the carry after each combination
// f(input[i], carry_in) must return the pair (result[i], carry_out)
template<tuple_like T, class C, class F>
constexpr auto inclusive_scan_and_fold(const T& input, const C& carry_in, const F& f)
{
  using namespace std;

  return tuples::fold_left(pair(tuple(), carry_in), input, [&f](const auto& prev_state, const auto& input_i)
  {
    // unpack the result of the previous fold iteration
    auto [prev_result, prev_carry] = prev_state;
    
    // combine the carry from the previous iteration with the current input element
    auto [result_i, carry_out] = f(input_i, prev_carry);

    // return the result of this iteration and the carry for the next iteration
    return pair(tuples::append_like<T>(prev_result, result_i), carry_out);
  });
}


template<class T>
concept tuple_of_pair_like =
  unzippable_tuple_like<T>
  and pair_like<element_t<0,T>>
;

template<class T>
concept pair_of_not_tuple_like =
  pair_like<T>
  and (not tuple_like<element_t<0,T>>)
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
  return tuples::unzip(tuple_of_pairs);
}

template<tuple_like T>
  requires (not tuple_of_pair_like<T> and not pair_of_not_tuple_like<T>)
constexpr pair_like auto unzip_innermost_pairs(const T& tuple)
{
  // this will return a pair of tuples
  return tuples::unzip(tuples::zip_with(tuple, [](auto t_i)
  {
    return tuples::unzip_innermost_pairs(t_i);
  }));
}


namespace detail
{


template<tuple_like T, class F, std::size_t... I>
constexpr auto unpack_and_invoke_impl(T&& arg, F&& f, std::index_sequence<I...>)
{
  return std::invoke(std::forward<F>(f), get<I>(std::forward<T>(arg))...);
}


} // end detail


template<tuple_like T, class F>
constexpr auto unpack_and_invoke(T&& args, F&& f)
{
  return detail::unpack_and_invoke_impl(std::forward<T>(args), std::forward<F>(f), indices_v<T>);
}


namespace detail
{


template<tuple_like R, tuple_like T, class F, std::size_t... I>
constexpr tuple_like auto static_enumerate_like_impl(const T& tuple, F&& f, std::index_sequence<I...>)
{
  return tuples::make_like<R>(f.template operator()<I>(get<I>(tuple))...); 
}


} // end detail


template<tuple_like R, tuple_like T, class F>
constexpr tuple_like auto static_enumerate_like(const T& tuple, F&& f)
{
  return detail::static_enumerate_like_impl<R>(tuple, std::forward<F>(f), indices_v<T>);
}

template<tuple_like T, class F>
constexpr tuple_like auto static_enumerate(const T& tuple, F&& f)
{
  return tuples::static_enumerate_like<T>(tuple, std::forward<F>(f));
}


template<std::size_t I, tuple_like T, class U>
  requires is_index_for<I,T>
constexpr tuple_like auto replace_element(const T& tuple, const U& replacement)
{
  return tuples::static_enumerate(tuple, [&]<std::size_t index>(const auto& t_i)
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


namespace detail
{

template<class F, tuple_like T, std::size_t... I>
constexpr bool invocable_on_all_elements_impl(std::index_sequence<I...>)
{
  return (... and invocable_on_element<F,I,T>);
}

} // end detail



template<class F, class T>
concept invocable_on_all_elements_of =
  tuple_like<T>
  and detail::invocable_on_all_elements_impl<F,T>(indices_v<T>)
;


template<tuple_like T, invocable_on_all_elements_of<T> F>
constexpr void for_each(T&& t, F&& f)
{
  auto invoke_on_each_element = [&]<std::size_t... I>(std::index_sequence<I...>)
  {
    constexpr auto sink_for_pack_expression = [](auto&&...){};

    // the business with the trailing comma zero avoids problems with void-returning f
    sink_for_pack_expression(
      (std::invoke(std::forward<F>(f), get<I>(std::forward<T>(t))), 0)...
    );
  };

  invoke_on_each_element(indices_v<T>);
}


template<class F, class... Args>
concept invocable_on_each = (... and std::invocable<F,Args>);

template<class F, class... Args>
  requires invocable_on_each<F&&,Args&&...>
constexpr void for_each_arg(F&& f, Args&&... args)
{
  tuples::for_each(std::forward_as_tuple(std::forward<Args>(args)...), std::forward<F>(f));
}


template<std::size_t N, tuple_like T>
  requires (N <= size_v<T>)
constexpr pair_like auto split_at(const T& t)
{
  return std::pair(tuples::take<N>(t), tuples::drop<N>(t));
}


} // end ubu::tuples


#include "../detail/epilogue.hpp"

