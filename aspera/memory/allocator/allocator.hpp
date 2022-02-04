#pragma once

#include "../../detail/prologue.hpp"

#include "allocate.hpp"
#include "deallocate.hpp"
#include <concepts>
#include <memory>

ASPERA_NAMESPACE_OPEN_BRACE

template<class A>
concept allocator = 
  std::equality_comparable<A> and
  std::is_nothrow_copy_constructible_v<A> and
  std::is_nothrow_destructible_v<A> and
  requires { typename std::decay_t<A>::value_type; } and

  requires(std::decay_t<A>& a, typename std::allocator_traits<std::decay_t<A>>::size_type n, typename std::allocator_traits<std::decay_t<A>>::pointer p)
  {
    ASPERA_NAMESPACE::allocate(a, n);
    ASPERA_NAMESPACE::deallocate(a, p, n);
  }
;

ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

