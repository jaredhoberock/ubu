#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/coordinates/concepts/coordinate.hpp"
#include "../allocate.hpp"
#include "../concepts/allocator.hpp"
#include "allocator_shape.hpp"
#include "allocator_value.hpp"
#include <type_traits>

namespace ubu
{


template<allocator A, class T = allocator_value_t<A>, coordinate S = allocator_shape_t<A>>
using allocator_view_t = allocate_result_t<T, std::remove_cvref_t<A>&, S>;


} // end ubu

#include "../../../../detail/epilogue.hpp"

