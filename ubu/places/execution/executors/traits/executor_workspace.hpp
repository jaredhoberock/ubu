#pragma once

#include "../../../../detail/prologue.hpp"

#include <cstddef>
#include <span>
#include <type_traits>

namespace ubu
{
namespace detail
{


// by default, E's workspace type is std::span<std::byte>
template<class E, class A>
struct executor_workspace
{
  using type = std::span<std::byte>;
};


template<class E>
concept has_member_workspace_type = requires
{
  typename E::workspace_type;
};

// if E has a member type ::workspace_type, use it
template<class E, class A>
  requires has_member_workspace_type<E>
struct executor_workspace<E,A>
{
  using type = typename E::workspace_type;
};


template<class E, class A>
concept has_member_workspace_type_template = requires
{
  typename E::template workspace_type<A>;
};

template<class E, class A>
  requires (not has_member_workspace_type<E>
            and not std::is_void_v<A>
            and has_member_workspace_type_template<E,A>)
struct executor_workspace<E,A>
{
  using type = E::template workspace_type<A>;
};

} // end detail

template<executor E, class A = void>
using executor_workspace_t = typename detail::executor_workspace<std::remove_cvref_t<E>,std::remove_cvref_t<A>>::type;

} // end ubu

#include "../../../../detail/epilogue.hpp"

