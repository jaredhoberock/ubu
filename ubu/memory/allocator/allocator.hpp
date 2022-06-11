#pragma once

#include "../../detail/prologue.hpp"

#include "allocate.hpp"
#include "deallocate.hpp"
#include <concepts>
#include <memory>

namespace ubu
{


template<class A, class T>
concept allocator_of =
  std::equality_comparable<A>
  and std::is_nothrow_destructible_v<A>
  and std::is_nothrow_destructible_v<A>
  and requires { typename std::decay_t<A>::value_type; }

  and requires(std::decay_t<A>& a, typename std::allocator_traits<std::decay_t<A>>::size_type n)
  {
    ubu::deallocate(a, ubu::allocate<T>(a, n), n);
  }
;


template<class A>
concept allocator = allocator_of<A, int>;


} // end ubu

#include "../../detail/epilogue.hpp"

