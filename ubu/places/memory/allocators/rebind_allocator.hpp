#pragma once

#include "../../../detail/prologue.hpp"
#include "../../../miscellaneous/detail/instantiatable.hpp"

#include <memory>
#include <utility>

namespace ubu
{
namespace detail
{

template<class A, class T>
concept has_rebind_alloc_member_template = requires
{
  typename A::template rebind_alloc<T>; 
};

template<class A, class T>
struct is_first_template_parameter_rebindable_with : std::false_type {};

template<template<class...> class Template, class T, class FirstArg, class... Args>
struct is_first_template_parameter_rebindable_with<Template<FirstArg,Args...>,T> : std::integral_constant<bool, instantiatable<Template,T,Args...>> {};

template<class A, class T>
concept has_allocator_traits_rebind_alloc =
  has_rebind_alloc_member_template<A,T>
  or is_first_template_parameter_rebindable_with<A,T>::value
;

} // end detail


template<class T, class A>
  requires detail::has_allocator_traits_rebind_alloc<std::remove_cvref_t<A>, T>
auto rebind_allocator(A&& alloc)
{
  typename std::allocator_traits<std::remove_cvref_t<A>>::template rebind_alloc<T> result{std::forward<A>(alloc)};
  return result;
}

template<class T, class A>
using rebind_allocator_result_t = decltype(ubu::rebind_allocator<T>(std::declval<A>()));


namespace detail
{


template<class T, class A>
concept has_rebind_allocator = requires(A alloc)
{
  rebind_allocator<T>(alloc);
};


} // end detail


} // end ubu

#include "../../../detail/epilogue.hpp"

