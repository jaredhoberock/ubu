#pragma once

#include "../../detail/prologue.hpp"
#include "../../grid/coordinate/detail/tuple_algorithm.hpp"
#include "../../memory/buffer/get_buffer.hpp"
#include "workspace.hpp"
#include <ranges>
#include <utility>

namespace ubu
{


// XXX shape is not the best thing to call this because a workspace is not a grid
template<workspace W>
inline constexpr auto workspace_shape(W ws)
{
  return std::ranges::size(get_buffer(ws));
}

template<hierarchical_workspace W>
inline constexpr auto workspace_shape(W ws)
{
  return detail::tuple_append(detail::ensure_tuple(workspace_shape(get_local_workspace(ws)), std::ranges::size(get_buffer(ws))));
}

template<workspace W>
using workspace_shape_t = decltype(workspace_shape(std::declval<W>()));


} // end ubu

#include "../../detail/epilogue.hpp"


