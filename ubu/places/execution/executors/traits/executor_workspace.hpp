#pragma once

#include "../../../../detail/prologue.hpp"

#include <cstddef>
#include <span>
#include <type_traits>

namespace ubu
{
namespace detail
{

template<class E>
struct executor_workspace
{
  using type = std::span<std::byte>;
};

template<class E>
  requires requires { typename std::remove_cvref_t<E>::workspace_type; }
struct executor_workspace<E>
{
  using type = typename std::remove_cvref_t<E>::workspace_type;
};

} // end detail

template<executor E>
using executor_workspace_t = typename detail::executor_workspace<E>::type;

} // end ubu

#include "../../../../detail/epilogue.hpp"

