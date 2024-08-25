#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../cooperators/workspaces/workspace_shape.hpp"
#include "executor_workspace.hpp"
#include <type_traits>

namespace ubu
{
namespace detail
{


// by default, we retrieve an executor's workspace_shape from executor_workspace_t
template<class E, class A>
struct executor_workspace_shape
{
  using type = workspace_shape_t<executor_workspace_t<E,A>>;
};


template<class E>
concept has_member_workspace_shape_type = requires
{
  typename E::workspace_shape;
};

// if E has a member type ::workspace_shape_type, use it
template<class E, class A>
  requires has_member_workspace_shape_type<E>
struct executor_workspace_shape<E,A>
{
  using type = typename E::workspace_shape_type;
};



template<class E, class A>
concept has_member_workspace_shape_type_template = requires
{
  typename E::template workspace_shape_type<A>;
};


// if E has no member type ::workspace_shape_type,
// and A is not void,
// and E has a member template ::workspace_shape_type<A>, use it
template<class E, class A>
  requires (not has_member_workspace_shape_type<E>
            and not std::is_void_v<A>
            and has_member_workspace_shape_type_template<E,A>)
struct executor_workspace_shape<E,A>
{
  using type = E::template workspace_shape_type<A>;
};


} // end detail

template<executor E, class A = void>
using executor_workspace_shape_t = typename detail::executor_workspace_shape<std::remove_cvref_t<E>,std::remove_cvref_t<A>>::type;

} // end ubu

#include "../../../../detail/epilogue.hpp"

