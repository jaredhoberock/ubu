#pragma once

#include "../../detail/prologue.hpp"

#include <memory>
#include <utility>

ASPERA_NAMESPACE_OPEN_BRACE


template<class T, class A>
typename std::allocator_traits<std::decay_t<A>>::template rebind_alloc<T> rebind_allocator(A&& alloc)
{
  typename std::allocator_traits<std::decay_t<A>>::template rebind_alloc<T> result{std::forward<A>(alloc)};
  return result;
}

template<class T, class A>
using rebind_allocator_result_t = decltype(ASPERA_NAMESPACE::rebind_allocator<T>(std::declval<A>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

