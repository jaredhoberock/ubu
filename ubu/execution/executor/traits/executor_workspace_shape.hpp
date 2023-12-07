#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../cooperation/workspace/workspace_shape.hpp"
#include "executor_workspace.hpp"
#include <type_traits>

namespace ubu
{
namespace detail
{

template<class E>
struct executor_workspace_shape
{
  using type = workspace_shape_t<executor_workspace_t<E>>;
};

template<class E>
  requires requires { typename std::remove_cvref_t<E>::workspace_shape_type; }
struct executor_workspace_shape<E>
{
  using type = typename std::remove_cvref_t<E>::workspace_shape_type;
};

} // end detail

template<executor E>
using executor_workspace_shape_t = typename detail::executor_workspace_shape<E>::type;

} // end ubu

#include "../../../detail/epilogue.hpp"

