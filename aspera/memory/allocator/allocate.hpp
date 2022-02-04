#pragma once

#include "../../detail/prologue.hpp"

#include <memory>
#include <utility>

ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


// this is the type of allocate
struct dispatch_allocate
{
  // dispatch just uses std::allocator_traits
  template<class Alloc, class N>
  constexpr auto operator()(Alloc&& alloc, N&& n) const
    -> decltype(std::allocator_traits<std::decay_t<Alloc>>::allocate(std::forward<Alloc>(alloc), std::forward<N>(n)))
  {
    return std::allocator_traits<std::decay_t<Alloc>>::allocate(std::forward<Alloc>(alloc), std::forward<N>(n));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_allocate allocate;

} // end anonymous namespace


template<class A, class N>
using allocate_result_t = decltype(ASPERA_NAMESPACE::allocate(std::declval<A>(), std::declval<N>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

