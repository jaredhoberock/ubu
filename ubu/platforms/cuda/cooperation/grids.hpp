#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../cooperation/primitives/concepts/cooperator.hpp"
#include "../../../cooperation/primitives/traits/cooperator_thread_scope.hpp"
#include "detail/sync_grid_count_half.hpp"
#include <nv/target>


namespace ubu::cuda
{


template<class C>
concept cooperative_grid_like =
  cooperator<C>
  and cooperator_thread_scope_v<C> == "device"
;


template<cooperative_grid_like G>
constexpr std::uint16_t synchronize_and_count(G, bool value)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (
    return detail::sync_grid_count_half(value);
  ), (
    assert(false);
    return 0;
  ))
}


} // end ubu::cuda

#include "../../../detail/epilogue.hpp"

