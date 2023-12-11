#include <algorithm>
#include <memory>
#include <ranges>
#include <ubu/causality/past_event.hpp>
#include <ubu/causality/wait.hpp>
#include <ubu/execution/executor/bulk_execute_with_workspace_after.hpp>
#include <ubu/grid/coordinate/point.hpp>
#include <ubu/grid/lattice.hpp>
#include <ubu/memory/buffer/reinterpret_buffer.hpp>
#include <ubu/platform/cpp/inline_executor.hpp>

#undef NDEBUG
#include <cassert>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>

template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif

namespace ns = ubu;


template<class T>
struct inline_allocator : std::allocator<T>
{
  using value_type = T;
  using happening_type = ns::past_event;

  auto allocate_after(happening_type, std::size_t n) const
  {
    return std::pair(ns::past_event(), std::allocator<T>().allocate(n));
  }

  happening_type deallocate_after(happening_type, T* ptr, std::size_t n) const
  {
    std::allocator<T>().deallocate(ptr, n);
    return {};
  }
};

static_assert(ns::asynchronous_allocator<inline_allocator<int>&>);


struct has_bulk_execute_with_workspace_after_member
{
  template<class F>
  ns::past_event bulk_execute_with_workspace_after(ns::past_event before, int n, int workspace_size, F&& f) const
  {
    before.wait();

    std::vector<std::byte> buffer(workspace_size);
    std::span workspace(buffer.data(), buffer.size());

    for(int i = 0; i < n; ++i)
    {
      f(i, workspace);
    }

    return {};
  }
};


struct has_bulk_execute_with_workspace_after_free_function {};

template<class F>
ns::past_event bulk_execute_with_workspace_after(const has_bulk_execute_with_workspace_after_free_function&, ns::past_event before, int n, int workspace_size, F&& f)
{
  before.wait();

  std::vector<std::byte> buffer(workspace_size);
  std::span workspace(buffer.data(), buffer.size());

  for(int i = 0; i < n; ++i)
  {
    f(i, workspace);
  }

  return {};
}


template<std::ranges::view V, ns::coordinate S>
constexpr bool are_ascending_coordinates(V coords, S shape)
{
  ns::lattice expected(shape);

  for(int i = 0; i != coords.size(); ++i)
  {
    if(expected[i] != coords[i]) return false;
  }

  return true;
}


void test()
{
  inline_allocator<std::byte> alloc;

  {
    has_bulk_execute_with_workspace_after_member e;

    int counter = 0;

    ns::past_event before;
    int expected = 10;

    auto done = ns::bulk_execute_with_workspace_after(e, alloc, before, expected, expected * sizeof(int), [&](int coord, auto workspace)
    { 
      auto coords = ns::reinterpret_buffer<int>(workspace);
      coords[coord] = coord;

      if(++counter == expected)
      {
        assert(are_ascending_coordinates(coords, expected));
      }
    });
    ns::wait(done);

    assert(expected == counter);
  }

  {
    has_bulk_execute_with_workspace_after_free_function e;

    int counter = 0;

    ns::past_event before;
    int expected = 10;

    auto done = ns::bulk_execute_with_workspace_after(e, alloc, before, expected, expected * sizeof(int), [&](int coord, auto workspace)
    {
      auto coords = ns::reinterpret_buffer<int>(workspace);
      coords[coord] = coord;

      if(++counter == expected)
      {
        assert(are_ascending_coordinates(coords, expected));
      }
    });
    ns::wait(done);

    assert(expected == counter);
  }

  {
    // 1D grid shape with inline_executor

    ns::cpp::inline_executor e;

    int counter = 0;

    ns::past_event before;
    int expected = 10;

    auto done = ns::bulk_execute_with_workspace_after(e, alloc, before, expected, expected * sizeof(int), [&](int coord, auto workspace)
    {
      auto coords = ns::reinterpret_buffer<int>(workspace);
      coords[coord] = coord;

      if(++counter == expected)
      {
        assert(are_ascending_coordinates(coords, expected));
      }
    });
    ns::wait(done);

    assert(expected == counter);
  }

  {
    // 2D grid shape with inline_executor

    ns::cpp::inline_executor e;

    int counter = 0;

    ns::past_event before;

    ns::int2 grid_shape{2,5};
    int n = ns::shape_size(grid_shape);

    auto done = ns::bulk_execute_with_workspace_after(e, alloc, before, grid_shape, n * sizeof(ns::int2), [&](ns::int2 coord, auto workspace)
    { 
      auto coords = ns::reinterpret_buffer<ns::int2>(workspace);
      coords[counter] = coord;

      if(++counter == n)
      {
        assert(are_ascending_coordinates(coords, grid_shape));
      }
    });
    ns::wait(done);

    assert(grid_shape.product() == counter);
  }

  {
    // 3D grid space with inline_executor

    ns::cpp::inline_executor e;

    int counter = 0;

    ns::past_event before;

    ns::int3 grid_shape{2,5,7};
    int n = ns::shape_size(grid_shape);

    auto done = ns::bulk_execute_with_workspace_after(e, alloc, before, grid_shape, n * sizeof(ns::int3), [&](ns::int3 coord, auto workspace)
    { 
      auto coords = ns::reinterpret_buffer<ns::int3>(workspace);
      coords[counter] = coord;

      if(++counter == n)
      {
        assert(are_ascending_coordinates(coords, grid_shape));
      }
    });
    ns::wait(done);

    assert(grid_shape.product() == counter);
  }
}

void test_bulk_execute_with_workspace_after()
{
  test();

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] ()
  {
    test();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}


