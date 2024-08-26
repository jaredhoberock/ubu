#pragma once

#include "../../../../../detail/prologue.hpp"

#include <memory>
#include <type_traits>

namespace ubu::detail
{


// by default, A's shape type is std::size_t
template<class A>
struct maybe_allocator_shape
{
  using type = std::size_t;
};


template<class A>
concept has_member_shape_type = requires
{
  typename std::remove_cvref_t<A>::shape_type;
};


// if A has a member type ::shape_type, use it
template<class A>
  requires has_member_shape_type<A>
struct maybe_allocator_shape<A>
{
  using type = typename std::remove_cvref_t<A>::shape_type;
};

template<class A>
concept has_allocator_traits_size_type = requires
{
  typename std::allocator_traits<std::remove_cvref_t<A>>::size_type;
};

// if std::allocator_traits<A>::size_type exists, use it
template<class A>
  requires (not has_member_shape_type<A>
            and has_allocator_traits_size_type<A>)
struct maybe_allocator_shape<A>
{
  using type = typename std::allocator_traits<std::remove_cvref_t<A>>::size_type;
};


// the difference between this and allocator_shape_t<A>
// is that A needn't be an allocator to use this trait
template<class A>
using maybe_allocator_shape_t = typename maybe_allocator_shape<A>::type;


} // end ubu::detail

#include "../../../../../detail/epilogue.hpp"

