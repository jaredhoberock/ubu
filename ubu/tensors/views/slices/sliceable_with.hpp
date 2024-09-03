#pragma once

#include "../../../detail/prologue.hpp"

#include "../../concepts/viewable_tensor.hpp"
#include "slicer.hpp"
#include <utility>

namespace ubu
{

template<class T, class K>
concept sliceable_with =
  viewable_tensor<T>
  and slicer_for<K,tensor_shape_t<T>>
;

} // end ubu


#include "../../../detail/epilogue.hpp"

