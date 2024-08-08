#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../causality/asynchronous_view_of.hpp"
#include <utility>

namespace ubu::detail
{

// the purpose of custom_allocate_and_zero_after is to find an
// (allocator, executor) pair's customization of allocate_and_zero_after
//
// the ways to customize this function are all subtly different
// half of these use both the allocator and executor parameters, and
// the other half ignores the executor parameter
//
// dispatch looks for a customization of allocate_and_zero_after,
// which are, in decreasing priority:
//
// these are customizations that use the executor parameter
// 0. alloc.template allocate_and_zero_after<T>(exec, before, n)
// 1. allocate_and_zero_after<T>(exec, before, n)
// 2. alloc.allocate_and_zero_after(exec, before, n)
// 3. allocate_and_zero_after(alloc, exec, before, n)
//
// and these are customizations that ignore the executor parameter 
// 4. alloc.template allocate_and_zero_after<T>(before, n)
// 5. allocate_and_zero_after<T>(before, n)
// 6. alloc.allocate_and_zero_after(before, n)
// 7. allocate_and_zero_after(alloc, before, n)

template<class T, class A, class E, class B, class S>
concept has_allocate_and_zero_after_customization_0 = requires(A alloc, E exec, B before, S shape)
{
  { alloc.template allocate_and_zero_after<T>(exec, before, shape) } -> asynchronous_view_of<T,S>;
};

template<class T, class A, class E, class B , class S>
concept has_allocate_and_zero_after_customization_1 = requires(A alloc, E exec, B before, S shape)
{
  { allocate_and_zero_after<T>(alloc, exec, before, shape) } -> asynchronous_view_of<T,S>;
};

template<class T, class A, class E, class B, class S>
concept has_allocate_and_zero_after_customization_2 = requires(A alloc, E exec, B before, S shape)
{
  { alloc.allocate_and_zero_after(exec, before, shape) } -> asynchronous_view_of<T,S>;
};

template<class T, class A, class E, class B, class S>
concept has_allocate_and_zero_after_customization_3 = requires(A alloc, E exec, B before, S shape)
{
  { allocate_and_zero_after(alloc, exec, before, shape) } -> asynchronous_view_of<T,S>;
};

template<class T, class A, class E, class B, class S>
concept has_allocate_and_zero_after_customization_4 = requires(A alloc, B before, S shape)
{
  { alloc.template allocate_and_zero_after<T>(before, shape) } -> asynchronous_view_of<T,S>;
};

template<class T, class A, class E, class B , class S>
concept has_allocate_and_zero_after_customization_5 = requires(A alloc, B before, S shape)
{
  { allocate_and_zero_after<T>(alloc, before, shape) } -> asynchronous_view_of<T,S>;
};

template<class T, class A, class E, class B, class S>
concept has_allocate_and_zero_after_customization_6 = requires(A alloc, B before, S shape)
{
  { alloc.allocate_and_zero_after(before, shape) } -> asynchronous_view_of<T,S>;
};

template<class T, class A, class E, class B, class S>
concept has_allocate_and_zero_after_customization_7 = requires(A alloc, B before, S shape)
{
  { allocate_and_zero_after(alloc, before, shape) } -> asynchronous_view_of<T,S>;
};


template<class T, class A, class E, class B, class S>
concept has_allocate_and_zero_after_customization =
  has_allocate_and_zero_after_customization_0<T,A,E,B,S> or
  has_allocate_and_zero_after_customization_1<T,A,E,B,S> or
  has_allocate_and_zero_after_customization_2<T,A,E,B,S> or
  has_allocate_and_zero_after_customization_3<T,A,E,B,S> or
  has_allocate_and_zero_after_customization_4<T,A,E,B,S> or
  has_allocate_and_zero_after_customization_5<T,A,E,B,S> or
  has_allocate_and_zero_after_customization_6<T,A,E,B,S> or
  has_allocate_and_zero_after_customization_7<T,A,E,B,S>
;


template<class T, class A, class E, class B, class S>
  requires has_allocate_and_zero_after_customization<T,A&&,E&&,B&&,S&&>
constexpr asynchronous_view_of<T,S> auto custom_allocate_and_zero_after(A&& alloc, E&& exec, B&& before, S&& shape)
{
  if constexpr (has_allocate_and_zero_after_customization_0<T,A&&,E&&,B&&,S&&>)
  {
    return std::forward<A>(alloc).template allocate_and_zero_after<T>(std::forward<E>(exec), std::forward<B>(before), std::forward<S>(shape));
  }
  else if constexpr (has_allocate_and_zero_after_customization_1<T,A&&,E&&,B&&,S&&>)
  {
    return allocate_and_zero_after<T>(std::forward<A>(alloc), std::forward<E>(exec), std::forward<B>(before), std::forward<S>(shape));
  }
  else if constexpr (has_allocate_and_zero_after_customization_2<T,A&&,E&&,B&&,S&&>)
  {
    return std::forward<A>(alloc).allocate_and_zero_after(std::forward<E>(exec), std::forward<B>(before), std::forward<S>(shape));
  }
  else if constexpr (has_allocate_and_zero_after_customization_3<T,A&&,E&&,B&&,S&&>)
  {
    return allocate_and_zero_after(std::forward<A>(alloc), std::forward<E>(exec), std::forward<B>(before), std::forward<S>(shape));
  }
  else if constexpr (has_allocate_and_zero_after_customization_4<T,A&&,E&&,B&&,S&&>)
  {
    return std::forward<A>(alloc).template allocate_and_zero_after<T>(std::forward<B>(before), std::forward<S>(shape));
  }
  else if constexpr (has_allocate_and_zero_after_customization_5<T,A&&,E&&,B&&,S&&>)
  {
    return allocate_and_zero_after<T>(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape));
  }
  else if constexpr (has_allocate_and_zero_after_customization_6<T,A&&,E&&,B&&,S&&>)
  {
    return std::forward<A>(alloc).allocate_and_zero_after(std::forward<B>(before), std::forward<S>(shape));
  }
  else
  {
    return allocate_and_zero_after(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape));
  }
}


template<class T, class A, class E, class B, class S>
concept has_custom_allocate_and_zero_after = requires(A alloc, E exec, B before, S shape)
{
  custom_allocate_and_zero_after<T>(std::forward<A>(alloc), std::forward<E>(exec), std::forward<B>(before), std::forward<S>(shape));
};
 

} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

