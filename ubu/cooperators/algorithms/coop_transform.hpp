#pragma once

#include "../../detail/prologue.hpp"

#include "../../tensors/traits/tensor_reference.hpp"
#include "../../tensors/concepts/tensor.hpp"
#include "../../tensors/views/zip.hpp"
#include "../concepts/cooperator.hpp"
#include <concepts>

namespace ubu
{

template<cooperator C, tensor I, tensor O, std::invocable<tensor_reference_t<I>> F>
  requires std::assignable_from<tensor_reference_t<O>, tensor_reference_t<I>>
constexpr void coop_transform(C self, I input, O output, F f)
{
  coop_for_each(self, zip(output, input), [f](auto out_and_in)
  {
    auto&& [out,in] = out_and_in;

    // XXX do we need forward on these?
    out = f(in);
  });
}

} // end ubu

#include "../../detail/epilogue.hpp"

