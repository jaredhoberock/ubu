#pragma once

#include "../../detail/prologue.hpp"
#include "../traits/tensor_rank.hpp"
#include "tensor_like.hpp"
#include "tensor_like_of_rank.hpp"
#include <concepts>

namespace ubu
{


// XXX this should be variadic
template<class A, class B>
concept same_tensor_rank =
  tensor_like<A>
  and tensor_like_of_rank<B, tensor_rank_v<A>>
;

} // end ubu

#include "../../detail/epilogue.hpp"

