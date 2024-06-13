#pragma once

#include "../../../detail/prologue.hpp"

#include "../pointers.hpp"
#include "rebind_allocator.hpp"
#include <memory>
#include <type_traits>
#include <utility>

namespace ubu
{

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

  // this path first rebind_allocators and then recurses
  template<class Alloc, pointer_like P, class N>
    requires (not has_allocator_traits_deallocate<Alloc&&,P,N&&>
              and has_rebind_allocator<pointer_pointee_t<P>,Alloc&&>)
              and has_allocator_traits_deallocate<
                rebind_allocator_result_t<pointer_pointee_t<P>,Alloc&&>,
                P, N&&
              >
  constexpr decltype(auto) operator()(Alloc&& alloc, P p, N&& n) const
  {
    auto rebound_alloc = rebind_allocator<pointer_pointee_t<P>>(std::forward<Alloc>(alloc));
    return (*this)(rebound_alloc, std::forward<P>(p), std::forward<N>(n));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_deallocate deallocate;

} // end anonymous namespace


template<class A, class P, class N>
using deallocate_result_t = decltype(ubu::deallocate(std::declval<A>(), std::declval<P>(), std::declval<N>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

