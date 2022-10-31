// ~/Desktop/circle/circle -std=c++20 -I. --cuda-path=/usr/local/cuda -sm_60 --verbose new_shmalloc_reduce.cpp -o a.out

#include "measure_bandwidth_of_invocation.hpp"
#include "shmalloc_reduce_kernel.hpp"
#include "tile.hpp"
#include <ubu/execution/executor.hpp>
#include <ubu/memory/allocator.hpp>
#include <ubu/platform/cuda/device_allocator.hpp>
#include <ubu/platform/cuda/kernel_executor.hpp>
#include <ubu/platform/cuda/managed_allocator.hpp>
#include <concepts>
#include <random>
#include <vector>


template<std::random_access_iterator I, std::random_access_iterator O, binary_invocable<std::iter_value_t<I>> F>
  requires plain_old_reducible<I,O,F>
ubu::cuda::event reduce_after_to(ubu::cuda::kernel_executor ex, ubu::cuda::device_allocator<int> alloc, const ubu::cuda::event& before, I first, int n, O result, F op)
{
  using namespace ubu;

  constexpr int block_size = 32*6;

  int num_multiprocessors = 0;
  ubu::cuda::detail::throw_on_error(
    cudaDeviceGetAttribute(&num_multiprocessors, cudaDevAttrMultiProcessorCount, ex.device()),
    "reduce_after_to: After cudaDeviceGetAttribute"
  );

  // 695 was determined empirically
  int max_num_ctas = num_multiprocessors * 695;

  // 11 was revealed in a dream
  int min_tile_size = block_size * 11;

  using namespace std::views;
  auto tiles = tile_evenly(counted(first,n), max_num_ctas, min_tile_size);

  int num_tiles = tiles.size();

  if(num_tiles > 1)
  {
    // allocate storage for each tile's reduction
    using partial_sum_type = std::invoke_result_t<F,std::iter_value_t<I>,std::iter_value_t<I>>;

    auto [allocation_ready, partial_results] = allocate_after<partial_sum_type>(alloc, before, num_tiles);

    // reduce each tile into a partial result
    auto first_phase = bulk_execute_after(ex, allocation_ready, {tiles.size(),block_size}, [=](ubu::int2 idx)
    {
      shmalloc_reduce_tiles_kernel<block_size>(idx.x, idx.y, tiles, partial_results, op);
    });

    // finish up in a second phase by reducing a single tile with a larger block

    auto single_tile_of_partial_results = tile(counted(partial_results, tiles.size()), tiles.size());
    auto second_phase = bulk_execute_after(ex, first_phase, 512, [=](int thread_idx)
    {
      shmalloc_reduce_tiles_kernel<512>(0, thread_idx, single_tile_of_partial_results, result, op);
    });

    // deallocate storage
    return deallocate_after(alloc, second_phase, partial_results, num_tiles);
  }

  // the input is small enough that it only requires a single phase 
  
  return bulk_execute_after(ex, before, 512, [=](int thread_idx)
  {
    shmalloc_reduce_tiles_kernel<512>(0, thread_idx, tiles, result, op);
  });
}


template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;


void test_correctness(int max_size)
{
  device_vector<int> input(max_size, 1);

  std::default_random_engine rng;
  for(int& x : input)
  {
    x = rng();
  }

  device_vector<int> result(1,0);
  device_vector<int> temporary(max_size);
  ubu::cuda::kernel_executor ex;
  ubu::cuda::device_allocator<int> alloc;
  ubu::cuda::event before = ubu::first_cause(ex);

  for(int size = 1000; size < max_size; size += size / 100)
  {
    reduce_after_to(ex, alloc, before, input.data(), size, result.data(), std::plus{}).wait();

    // host reduce using std::accumulate.
    int ref = std::accumulate(input.begin(), input.begin() + size, 0);

    if(result[0] != ref)
    {
      printf("reduce:           %d\n", result[0]);
      printf("std::accumulate:  %d\n", ref);
      printf("Error at size: %d\n", size);
      exit(1);
    }
  }
}


double test_performance(int size, int num_trials)
{
  device_vector<int> input(size, 1);
  device_vector<int> result(1);
  ubu::cuda::kernel_executor ex;
  ubu::cuda::device_allocator<int> alloc;
  ubu::cuda::event before = ubu::first_cause(ex);

  // warmup
  reduce_after_to(ex, alloc, before, input.data(), size, result.data(), std::plus{});

  return measure_bandwidth_of_invocation_in_gigabytes_per_second(num_trials, sizeof(int) * input.size(), [&]
  {
    reduce_after_to(ex, alloc, before, input.data(), size, result.data(), std::plus{});
  });
}


int main()
{
  std::cout << "Testing correctness... " << std::flush;
  test_correctness(23456789);
  std::cout << "Done." << std::endl;

  std::cout << "Testing performance... " << std::flush;
  double bandwidth = test_performance(1 << 30, 1000);
  std::cout << "Done." << std::endl;

  std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

  return 0; 
}
