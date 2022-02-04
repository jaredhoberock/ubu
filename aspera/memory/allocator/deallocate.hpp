#pragma once

#include "../../detail/prologue.hpp"

#include <memory>
#include <utility>

ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


// this is the type of deallocate
struct dispatch_deallocate
{
  // dispatch just uses std::allocator_traits
  template<class Alloc, class P, class N>
  constexpr auto operator()(Alloc&& alloc, P&& p, N&& n) const
    -> decltype(std::allocator_traits<std::decay_t<Alloc>>::deallocate(std::forward<Alloc>(alloc), std::forward<P>(p), std::forward<N>(n)))
  {
    return std::allocator_traits<std::decay_t<Alloc>>::deallocate(std::forward<Alloc>(alloc), std::forward<P>(p), std::forward<N>(n));
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

