#pragma once

#include "../../../../detail/prologue.hpp"

#include "../allocate.hpp"
#include "../deallocate.hpp"
#include <concepts>
#include <cstddef>
#include <memory>

namespace ubu
{
namespace detail
{


// by default, A's shape type is std::size_t
template<class A>
struct allocator_shape
{
  using type = std::size_t;
};


template<class A>
concept has_member_shape_type = requires
{
  typename A::shape_type;
};


// if A has a member type ::shape_type, use it
template<class A>
  requires has_member_shape_type<A>
struct allocator_shape<A>
{
  using type = typename A::shape_type;
};

template<class A>
concept has_allocator_traits_size_type = requires
{
  typename std::allocator_traits<A>::size_type;
};

// if std::allocator_traits<A>::size_type exists, use it
template<class A>
  requires has_allocator_traits_size_type<A>
struct allocator_shape<A>
{
  using type = typename std::allocator_traits<A>::size_type;
};



} // end detail


// XXX the shape type ought to be a parameter
template<class A, class T>
concept allocator_of =
  std::equality_comparable<A>
  and std::is_nothrow_destructible_v<A>
  and std::is_nothrow_destructible_v<A>

  // we must be able to figure out the element type of tensors allocated by A
  and requires { typename std::remove_cvref_t<A>::value_type; }

  // we must be able to figure out the shape of tensors allocated by A
  and requires { typename detail::allocator_shape<std::remove_cvref_t<A>>::type; }

  and requires(std::remove_cvref_t<A>& a, typename detail::allocator_shape<std::remove_cvref_t<A>>::type shape)
  {
    ubu::deallocate(a, ubu::allocate<T>(a, shape));
  }
;


template<class A>
concept allocator = allocator_of<A, std::byte>;


} // end ubu

#include "../../../../detail/epilogue.hpp"

