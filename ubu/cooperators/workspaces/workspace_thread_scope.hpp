#pragma once

#include "../../detail/prologue.hpp"

#include "../barriers/barrier_like.hpp"
#include "../thread_scope.hpp"
#include "workspace.hpp"
#include <string_view>

namespace ubu
{
namespace detail
{

template<workspace W>
constexpr std::string_view workspace_thread_scope()
{
  // if the workspace has a static thread scope, use it
  if constexpr (has_static_thread_scope<W>)
  {
    return thread_scope_v<W>;
  }

  // else, use the thread scope of its barrier, if it has one
  if constexpr (concurrent_workspace<W>)
  {
    if constexpr (detail::has_static_thread_scope<barrier_t<W>>)
    {
      return thread_scope_v<barrier_t<W>>;
    }
  }

  // the default is "system"
  return "system";
}

} // end detail

template<workspace W>
inline constexpr const std::string_view workspace_thread_scope_v = detail::workspace_thread_scope<W>();

} // end ubu

#include "../../detail/epilogue.hpp"

