#pragma once

#include "../../../detail/prologue.hpp"
#include "../../../places/memory/data.hpp"
#include "../../../tensors/vectors/span_like.hpp"
#include "../../../utilities/integrals/size.hpp"

namespace ubu::detail
{

template<class S, class N>
concept advanceable_span_like =
  span_like<S>
  and requires(S& s, N n)
  {
    s = S(data(s) + n, size(s) - n);
  }
;


// XXX ideally, we would make this work for any memory_view
template<integral_like N, advanceable_span_like<N> S>
constexpr void advance_data(S& s, N n)
{
  s = S(data(s) + n, size(s) - n);
}

} // end ubu::detail

#include "../../../detail/epilogue.hpp"

