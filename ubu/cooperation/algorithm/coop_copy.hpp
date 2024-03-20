#pragma once

#include "../../detail/prologue.hpp"

#include "../../tensor/traits/tensor_reference.hpp"
#include "../../tensor/concepts/tensor_like.hpp"
#include "../cooperator/concepts/cooperator.hpp"
#include "coop_transform.hpp"
#include <concepts>
#include <functional>

namespace ubu
{

template<cooperator C, tensor_like S, tensor_like D>
  requires std::assignable_from<tensor_reference_t<D>, tensor_reference_t<S>>
constexpr void coop_copy(C self, S source, D destination)
{
  coop_transform(self, source, destination, std::identity());
}

} // end ubu

#include "../../detail/epilogue.hpp"

