#pragma once

#include <cassert>
#include <ubu/cooperators/workspaces.hpp>
#include <ubu/tensors/coordinates/concepts/congruent.hpp>
#include <ubu/tensors/coordinates/concepts/coordinate.hpp>
#include <ubu/tensors/shapes/shape_size.hpp>
#include <ubu/tensors/views/layouts/strides.hpp>
#include <ubu/utilities/tuples.hpp>
#include <span>

// validate_workspace is a utility function that other CUDA executor tests use
// the workspace is assumed to have a buffer at every level of size equal to the group of threads at that level

template<ubu::coordinate C, ubu::congruent<C> S, ubu::workspace W>
constexpr void validate_workspace(C coord, S shape, W ws)
{
  using namespace ubu;

  auto buffer = get_buffer(ws);

  // the buffer's size must accomodate an int per thread
  assert(sizeof(int) * shape_size(shape) == size(buffer));

  if constexpr (concurrent_workspace<W>)
  {
    std::span<int> indices = reinterpret_buffer<int>(buffer);

    // each thread records its index in the buffer
    int idx = apply_stride(compact_left_major_stride(shape), coord);
    indices[idx] = idx;

    // the threads synchronize
    arrive_and_wait(get_barrier(ws));

    // the first thread checks that each thread's index was recorded in the buffer
    if(idx == 0)
    {
      int expected = 0;
      for(int idx : indices)
      {
        assert(expected == idx);
        ++expected;
      }
    }
  }
  else
  {
    // each thread checks that the buffer is initialized to 0
    for(std::byte b : buffer)
    {
      assert(b == std::byte(0));
    }
  }

  if constexpr (hierarchical_workspace<W>)
  {
    auto local_coord = tuples::drop_last_and_unwrap_single(coord);
    auto local_shape = tuples::drop_last_and_unwrap_single(shape);
    auto local_ws = get_local_workspace(ws);

    validate_workspace(local_coord, local_shape, local_ws);
  }
}

// define this function expected by the Makefile
inline void test_validate_workspace()
{
}

