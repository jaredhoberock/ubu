#pragma once

#include "../../detail/prologue.hpp"

#include <memory>
#include <utility>

namespace ubu
{


template<class T, class A>
typename std::allocator_traits<std::decay_t<A>>::template rebind_alloc<T> rebind_allocator(A&& alloc)
{
  typename std::allocator_traits<std::decay_t<A>>::template rebind_alloc<T> result{std::forward<A>(alloc)};
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

#include "../../detail/epilogue.hpp"

