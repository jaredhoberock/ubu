#pragma once

#include "../../detail/prologue.hpp"

#include "rebind_allocator.hpp"
#include <memory>
#include <type_traits>
#include <utility>

ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class A, class P, class N>
concept has_allocator_traits_deallocate = requires(A alloc, P p, N n)
{
  std::allocator_traits<std::decay_t<A>>::deallocate(alloc, p, n);
};


// this is the type of deallocate
struct dispatch_deallocate
{
  // dispatch just uses std::allocator_traits
  template<class Alloc, class P, class N>
    requires has_allocator_traits_deallocate<Alloc&&,P&&,N&&>
  constexpr void operator()(Alloc&& alloc, P&& p, N&& n) const
  {
    std::allocator_traits<std::decay_t<Alloc>>::deallocate(std::forward<Alloc>(alloc), std::forward<P>(p), std::forward<N>(n));
  }

  // this path attempts to first rebind_allocator and then recurse
  template<class Alloc, class P, class N>
    requires (!has_allocator_traits_deallocate<Alloc&&,P&&,N&&> and
              has_rebind_allocator<typename std::pointer_traits<std::remove_cvref_t<P>>::element_type,Alloc&&>)
  constexpr decltype(auto) operator()(Alloc&& alloc, P&& p, N&& n) const
  {
    auto rebound_alloc = rebind_allocator<typename std::pointer_traits<P>::element_type>(std::forward<Alloc>(alloc));
    return (*this)(rebound_alloc, std::forward<P>(p), std::forward<N>(n));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_deallocate deallocate;

} // end anonymous namespace


template<class A, class P, class N>
using deallocate_result_t = decltype(ASPERA_NAMESPACE::deallocate(std::declval<A>(), std::declval<P>(), std::declval<N>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

