#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../cooperators/concepts/cooperator.hpp"
#include "../../../../cooperators/traits/cooperator_thread_scope.hpp"
#include "../../../../places/memory/views/basic_buffer.hpp"
#include <cstddef>
#include <span>
#include <string_view>


namespace ubu::cuda
{


struct block_workspace
{
  basic_buffer<int> buffer;

  struct barrier_type
  {
    constexpr static const std::string_view thread_scope = "block";

    constexpr void arrive_and_wait() const
    {
#if defined(__CUDACC__)
      __syncthreads();
#endif
    }
  };

  barrier_type barrier;
};


template<class C>
concept block_like =
  cooperator<C>
  and cooperator_thread_scope_v<C> == "block"
;


} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

