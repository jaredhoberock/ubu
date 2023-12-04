#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../cooperation/workspace/workspace_shape.hpp"
#include "executor_workspace.hpp"

namespace ubu
{

template<executor E>
using executor_workspace_shape_t = workspace_shape_t<executor_workspace_t<E>>;

} // end ubu

#include "../../../detail/epilogue.hpp"



