#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/first_cause.hpp"
#include "allocate_after.hpp"
#include "traits/allocator_value_t.hpp"
#include <memory>
#include <type_traits>
#include <utility>

namespace ubu
{

namespace detail
{


template<class T, class A, class N>
concept has_first_allocate_member_function_template = requires(A alloc, N n)
{
  // XXX this should check that the result is a pair<happening,pointer>
  alloc.template first_allocate<T>(n);
};


template<class T, class A, class N>
concept has_first_allocate_free_function_template = requires(A alloc, N n)
{
  // XXX this should check that the result is a pair<happening,pointer>
  first_allocate<T>(alloc, n);
};


template<class T, class A, class N>
concept has_first_allocate_member_function = requires(A alloc, N n)
{
  requires std::same_as<T, typename std::allocator_traits<std::remove_cvref_t<A>>::value_type>;

  // XXX this should check that the result is a pair<happening,pointer>
  alloc.first_allocate(n);
};


template<class T, class A, class N>
concept has_first_allocate_free_function = requires(A alloc, N n)
{
  requires std::same_as<T, typename std::allocator_traits<std::remove_cvref_t<A>>::value_type>;

  // XXX this should check that the result is a pair<happening,pointer>
  first_allocate(alloc, n);
};


template<class T, class A, class N>
concept has_first_allocate_customization =
  has_first_allocate_member_function_template<T,A,N>
  or has_first_allocate_free_function_template<T,A,N>
  or has_first_allocate_member_function<T,A,N>
  or has_first_allocate_free_function<T,A,N>
;


// this is the type of first_allocate
template<class T>
struct dispatch_first_allocate
{
  // this path calls the member function template
  template<class Alloc, class N>
    requires has_first_allocate_member_function_template<T, Alloc&&, N&&>
  constexpr auto operator()(Alloc&& alloc, N&& n) const
  {
    return std::forward<Alloc>(alloc).template first_allocate<T>(std::forward<N>(n));
  }
  
  // this path calls the free function template
  template<class Alloc, class N>
    requires (!has_first_allocate_member_function_template<T, Alloc&&, N&&>
              and has_first_allocate_free_function_template<T, Alloc&&, N&&>)
  constexpr auto operator()(Alloc&& alloc, N&& n) const
  {
    return first_allocate<T>(std::forward<Alloc>(alloc), std::forward<N>(n));
  }

  // this path calls the member function
  template<class Alloc, class N>
    requires (!has_first_allocate_member_function_template<T, Alloc&&, N&&>
              and !has_first_allocate_free_function_template<T, Alloc&&, N&&>
              and has_first_allocate_member_function<T, Alloc&&, N&&>)
  constexpr auto operator()(Alloc&& alloc, N&& n) const
  {
    return std::forward<Alloc>(alloc).first_allocate(std::forward<N>(n));
  }

  // this path calls the free function
  template<class Alloc, class N>
    requires (!has_first_allocate_member_function_template<T, Alloc&&, N&&>
              and !has_first_allocate_free_function_template<T, Alloc&&, N&&>
              and !has_first_allocate_member_function<T, Alloc&&, N&&>
              and has_first_allocate_free_function<T, Alloc&&, N&&>)
  constexpr auto operator()(Alloc&& alloc, N&& n) const
  {
    return first_allocate(std::forward<Alloc>(alloc), std::forward<N>(n));
  }

  // this dispatch path tries to rebind and then call first_allocate again
  template<class Alloc, class N>
    requires (!has_first_allocate_member_function_template<T, Alloc&&, N&&>
              and !has_first_allocate_free_function_template<T, Alloc&&, N&&>
              and !has_first_allocate_member_function<T, Alloc&&, N&&>
              and !has_first_allocate_free_function<T, Alloc&&, N&&>
              and has_first_allocate_customization<T, rebind_allocator_result_t<T,Alloc&&>, N&&>)
  constexpr auto operator()(Alloc&& alloc, N&& n) const
  {
    auto rebound_alloc = rebind_allocator<T>(std::forward<Alloc>(alloc));
    return (*this)(rebound_alloc, std::forward<N>(n));
  }

  // this dispatch path calls allocate_after
  template<class Alloc, class N>
    requires (!has_first_allocate_member_function_template<T, Alloc&&, N&&>
              and !has_first_allocate_free_function_template<T, Alloc&&, N&&>
              and !has_first_allocate_member_function<T, Alloc&&, N&&>
              and !has_first_allocate_free_function<T, Alloc&&, N&&>
              and !has_first_allocate_customization<T, rebind_allocator_result_t<T,Alloc&&>, N&&>)
  constexpr auto operator()(Alloc&& alloc, N&& n) const ->
    decltype(allocate_after<T>(std::forward<Alloc>(alloc), first_cause(std::forward<Alloc>(alloc)), std::forward<N>(n)))
  {
    return allocate_after<T>(std::forward<Alloc>(alloc), first_cause(std::forward<Alloc>(alloc)), std::forward<N>(n));
  }
};


} // end detail


namespace
{

template<class T>
constexpr detail::dispatch_first_allocate<T> first_allocate;

} // end anonymous namespace


template<class T, class A, class N>
using first_allocate_result_t = decltype(first_allocate<T>(std::declval<A>(), std::declval<N>()));


} // end ubu

#include "../../detail/epilogue.hpp"

