#pragma once

#include "../../detail/prologue.hpp"
#include "../traits/tensor_reference.hpp"
#include "tensor_like.hpp"
#include <concepts>

namespace ubu
{

template<class F, class Tensor, class... Tensors>
concept elemental_invocable =
  (tensor_like<Tensor> and ... and tensor_like<Tensors>)
  and std::invocable<F, tensor_reference_t<Tensor>, tensor_reference_t<Tensors>...>
;

} // end ubu

#include "../../detail/epilogue.hpp"

