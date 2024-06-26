#pragma once

#include "../../../detail/prologue.hpp"

#include "rebind_allocator.hpp"
#include <memory>
#include <type_traits>
#include <utility>

namespace ubu
{

namespace detail
{


template<class T, class A, class E, class N>
concept has_allocate_after_member_function_template = requires(A alloc, E before, N n)
{
  // XXX this should check that the result is a pair<E,pointer>
  alloc.template allocate_after<T>(before, n);
};


template<class T, class A, class E , class N>
concept has_allocate_after_free_function_template = requires(A alloc, E before, N n)
{
  // XXX this should check that the result is a pair<E,pointer>
  allocate_after<T>(alloc, before, n);
};


template<class T, class A, class E, class N>
concept has_allocate_after_member_function = requires(A alloc, E before, N n)
{
  requires std::same_as<T, typename std::allocator_traits<std::remove_cvref_t<A>>::value_type>;

  // XXX this should check that the result is a pair<E,pointer>
  alloc.allocate_after(before, n);
};


template<class T, class A, class E, class N>
concept has_allocate_after_free_function = requires(A alloc, E before, N n)
{
  requires std::same_as<T, typename std::allocator_traits<std::remove_cvref_t<A>>::value_type>;

  // XXX this should check that the result is a pair<E,pointer>
  allocate_after(alloc, before, n);
};


// this is the type of allocate_after
template<class T>
struct dispatch_allocate_after
{
  // this dispatch path calls a member function template
  template<class Alloc, class Event, class N>
    requires has_allocate_after_member_function_template<T, Alloc&&, Event&&, N&&>
  constexpr auto operator()(Alloc&& alloc, Event&& before, N&& n) const
  {
    return std::forward<Alloc>(alloc).template allocate_after<T>(std::forward<Event>(before), std::forward<N>(n));
  }

  // this dispatch path calls a free function template
  template<class Alloc, class Event, class N>
    requires (!has_allocate_after_member_function_template<T, Alloc&&, Event&&, N&&>
              and has_allocate_after_free_function_template<T, Alloc&&, Event&&, N&&>)
  constexpr auto operator()(Alloc&& alloc, Event&& before, N&& n) const
  {
    return allocate_after<T>(std::forward<Alloc>(alloc), std::forward<Event>(before), std::forward<N>(n));
  }

  // this dispatch path calls the member function
  template<class Alloc, class Event, class N>
    requires (!has_allocate_after_member_function_template<T, Alloc&&, Event&&, N&&>
              and !has_allocate_after_free_function_template<T, Alloc&&, Event&&, N&&>
              and has_allocate_after_member_function<T, Alloc&&, Event&&, N&&>)
  constexpr auto operator()(Alloc&& alloc, Event&& before, N&& n) const
  {
    return std::forward<Alloc>(alloc).allocate_after(std::forward<Event>(before), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Alloc, class Event, class N>
    requires (!has_allocate_after_member_function_template<T, Alloc&&, Event&&, N&&>
              and !has_allocate_after_free_function_template<T, Alloc&&, Event&&, N&&>
              and !has_allocate_after_member_function<T, Alloc&&, Event&&, N&&>
              and has_allocate_after_free_function<T, Alloc&&, Event&&, N&&>)
  constexpr auto operator()(Alloc&& alloc, Event&& before, N&& n) const
  {
    return allocate_after(std::forward<Alloc>(alloc), std::forward<Event>(before), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Alloc, class Event, class N>
    requires (!has_allocate_after_member_function_template<T, Alloc&&, Event&&, N&&>
              and !has_allocate_after_free_function_template<T, Alloc&&, Event&&, N&&>
              and !has_allocate_after_member_function<T, Alloc&&, Event&&, N&&>
              and !has_allocate_after_free_function<T, Alloc&&, Event&&, N&&>
              and has_rebind_allocator<T, Alloc&&>)
  constexpr decltype(auto) operator()(Alloc&& alloc, Event&& before, N&& n) const
  {
    auto rebound_alloc = rebind_allocator<T>(std::forward<Alloc>(alloc));

    return (*this)(rebound_alloc, std::forward<Event>(before), std::forward<N>(n));
  }
};


} // end detail


namespace
{

template<class T>
constexpr detail::dispatch_allocate_after<T> allocate_after;

} // end anonymous namespace


template<class T, class A, class E, class N>
using allocate_after_result_t = decltype(ubu::allocate_after<T>(std::declval<A>(), std::declval<E>(), std::declval<N>()));


} // end ubu


#include "../../../detail/epilogue.hpp"

