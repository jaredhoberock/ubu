#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../cooperation/cooperator/concepts/cooperator.hpp"
#include "../../../../cooperation/cooperator/traits/cooperator_thread_scope.hpp"
#include "../../../../memory/buffer/empty_buffer.hpp"
#include "../../../../miscellaneous/constant.hpp"
#include <string_view>

namespace ubu::cuda
{


constexpr auto warp_size = 32_c;
constexpr auto warp_mask = 0xFFFFFFFF_c;


struct warp_workspace
{
  empty_buffer buffer;

  struct barrier_type
  {
    constexpr static const std::string_view thread_scope = "warp";

    inline void arrive_and_wait() const
    {
#if defined(__CUDACC__)
      __syncwarp();
#endif
    }
  };

  barrier_type barrier;
};


// XXX we may wish to require that size(warp_like) must equal warp_size
template<class C>
concept warp_like =
  cooperator<C>
  and cooperator_thread_scope_v<C> == "warp"
;

} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

