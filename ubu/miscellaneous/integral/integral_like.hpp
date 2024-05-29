#pragma once

#include "../../detail/prologue.hpp"
#include "detail/as_integral.hpp"
#include <concepts>
#include <utility>

namespace ubu
{
namespace detail
{


template<class B>
concept boolean_testable_impl = std::convertible_to<B,bool>;

template<class B>
concept boolean_testable =
  boolean_testable_impl<B>
  and requires(B&& b)
  {
    { !std::forward<B>(b) } -> detail::boolean_testable_impl;
  }
;


} // end detail


// XXX not sure if integral_like<T> should also require boolean_testable<T>
template<class T>
concept integral_like = requires(T i)
{
  detail::as_integral(i);

  // unary operators
  detail::as_integral(+i);
  detail::as_integral(-i);
  { !i } -> detail::boolean_testable;

  // binary operators
  { i == i } -> detail::boolean_testable;
  { i != i } -> detail::boolean_testable;
  { i  < i } -> detail::boolean_testable;
  { i <= i } -> detail::boolean_testable;
  { i  > i } -> detail::boolean_testable;
  { i >= i } -> detail::boolean_testable;

  // arithmetic operators
  detail::as_integral(i  + i);
  detail::as_integral(i  - i);
  detail::as_integral(i  * i);
  detail::as_integral(i  / i);
  detail::as_integral(i  % i);
  detail::as_integral(i  & i);
  detail::as_integral(i  | i);
  detail::as_integral(i  ^ i);
  detail::as_integral(i << i);
  detail::as_integral(i >> i);
};


} // end ubu

#include "../../detail/epilogue.hpp"

