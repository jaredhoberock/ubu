#pragma once

#include "../../../../detail/prologue.hpp"

#include "../concepts/executor.hpp"
#include "executor_workspace_shape.hpp"
#include <type_traits>

namespace ubu
{
namespace detail
{

template<class E>
struct executor_shape
{
  using type = executor_workspace_shape_t<E>;
};

template<class E>
  requires requires { typename std::remove_cvref_t<E>::shape_type; }
struct executor_shape<E>
{
  using type = typename std::remove_cvref_t<E>::shape_type;
};

} // end detail

template<executor E>
using executor_shape_t = typename detail::executor_shape<E>::type;

} // end ubu

#include "../../../../detail/epilogue.hpp"

