#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../cooperation/concepts/cooperator.hpp"
#include "../../../../cooperation/primitives/traits/cooperator_thread_scope.hpp"
#include <cstddef>
#include <span>
#include <string_view>


namespace ubu::cuda
{


struct block_workspace
{
  // XXX we should use small_span or similar with int size
  std::span<std::byte> buffer;

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

