#pragma once

#include "../../detail/prologue.hpp"

#include "rebind_allocator.hpp"
#include <memory>
#include <utility>

namespace ubu
{

namespace detail
{


template<class T, class A, class N>
concept has_allocate_member_function_template = requires(A alloc, N n)
{
  // XXX this should check that the result is a pointer
  alloc.template allocate<T>(n);
};


template<class T, class A, class N>
concept has_allocate_free_function_template = requires(A alloc, N n)
{
  // XXX this should check that the result is a pointer
  allocate<T>(alloc, n);
};


template<class T, class A, class N>
concept has_allocator_traits_allocate = requires(A alloc, N n)
{
  requires std::same_as<T, typename std::allocator_traits<std::decay_t<A>>::value_type>;

  std::allocator_traits<std::decay_t<A>>::allocate(alloc, n);
};


// this is the type of allocate
template<class T>
struct dispatch_allocate
{
  // this path uses alloc.allocate<T>(n) when it exists
  template<class Alloc, class N>
    requires has_allocate_member_function_template<T,Alloc&&,N&&>
  constexpr auto operator()(Alloc&& alloc, N&& n) const
  {
    return std::forward<Alloc>(alloc).template allocate<T>(std::forward<N>(n));
  }

  // this path uses allocate<T>(alloc, n) when it exists
  template<class Alloc, class N>
    requires (!has_allocate_member_function_template<T,Alloc&&,N&&>
              and has_allocate_free_function_template<T,Alloc&&,N&&>)
  constexpr auto operator()(Alloc&& alloc, N&& n) const
  {
    return allocate<T>(std::forward<Alloc>(alloc), std::forward<N>(n));
  }

  // this path uses uses std::allocator_traits when T matches value_type
  template<class Alloc, class N>
    requires (!has_allocate_member_function_template<T,Alloc&&,N&&>
              and !has_allocate_free_function_template<T,Alloc&&,N&&>
              and has_allocator_traits_allocate<T,Alloc&&,N&&>)
  constexpr auto operator()(Alloc&& alloc, N&& n) const
  {
    return std::allocator_traits<std::decay_t<Alloc>>::allocate(std::forward<Alloc>(alloc), std::forward<N>(n));
  }

  // this path attempts to first rebind_allocator and then recurse
  template<class Alloc, class N>
    requires (!has_allocate_member_function_template<T,Alloc&&,N&&>
              and !has_allocate_free_function_template<T,Alloc&&,N&&>
              and !has_allocator_traits_allocate<T,Alloc&&,N&&>
              and has_rebind_allocator<T,Alloc&&>)
  constexpr decltype(auto) operator()(Alloc&& alloc, N&& n) const
  {
    auto rebound_alloc = rebind_allocator<T>(std::forward<Alloc>(alloc));
    return (*this)(rebound_alloc, std::forward<N>(n));
  }
};


} // end detail


namespace
{

template<class T>
constexpr detail::dispatch_allocate<T> allocate;

} // end anonymous namespace


template<class T, class A, class N>
using allocate_result_t = decltype(ubu::allocate<T>(std::declval<A>(), std::declval<N>()));


} // end ubu

#include "../../detail/epilogue.hpp"

