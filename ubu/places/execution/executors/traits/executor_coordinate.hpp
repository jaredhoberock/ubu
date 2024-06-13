#pragma once

#include "../../../../detail/prologue.hpp"

#include "../concepts/executor.hpp"
#include "executor_shape.hpp"

namespace ubu
{
namespace detail
{

template<class E>
struct executor_coordinate
{
  using type = executor_shape_t<E>;
};

template<class E>
  requires requires { typename std::remove_cvref_t<E>::coordinate_type; }
struct executor_coordinate<E>
{
  using type = typename std::remove_cvref_t<E>::coordinate_type;
};

} // end detail

template<executor E>
using executor_coordinate_t = typename detail::executor_coordinate<E>::type;

} // end ubu

#include "../../../../detail/epilogue.hpp"

