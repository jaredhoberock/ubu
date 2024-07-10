#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/integrals/size.hpp"
#include "../../utilities/tuples.hpp"
#include "../../places/memory/buffers/get_buffer.hpp"
#include "../../tensors/coordinates/point.hpp"
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
  return tuples::append(tuples::ensure_tuple_like<size2>(workspace_shape(get_local_workspace(ws))), size(get_buffer(ws)));
}

template<workspace W>
using workspace_shape_t = decltype(workspace_shape(std::declval<W>()));


} // end ubu

#include "../../detail/epilogue.hpp"

