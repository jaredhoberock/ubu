#pragma once

#include "../../detail/prologue.hpp"
#include "../../memory/buffer/get_buffer.hpp"
#include "../../miscellaneous/integral/size.hpp"
#include "../../tensor/coordinate/detail/tuple_algorithm.hpp"
#include "../../tensor/coordinate/point.hpp"
#include "hierarchical_workspace.hpp"
#include "workspace.hpp"
#include <utility>

namespace ubu
{


// XXX shape is not the best thing to call this because a workspace is not a tensor_like
template<workspace W>
inline constexpr auto workspace_shape(W ws)
{
  return size(get_buffer(ws));
}

template<hierarchical_workspace W>
inline constexpr auto workspace_shape(W ws)
{
  return detail::tuple_append(detail::ensure_tuple_similar_to<size2>(workspace_shape(get_local_workspace(ws))), size(get_buffer(ws)));
}

template<workspace W>
using workspace_shape_t = decltype(workspace_shape(std::declval<W>()));


} // end ubu

#include "../../detail/epilogue.hpp"

