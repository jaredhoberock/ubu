// circle -std=c++20 -I../../../ --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/12.0/ -sm_60 --verbose reduce.cpp --libstdc++=11 -L/usr/local/cuda/lib64 -lcudart -o reduce.out

#include "measure_bandwidth_of_invocation.hpp"
#include "reduce_kernel.hpp"
#include "tile.hpp"
#include <ubu/causality/initial_happening.hpp>
#include <ubu/execution/executor/bulk_execute_after.hpp>
#include <ubu/memory/allocator.hpp>
#include <ubu/platform/cuda/device_allocator.hpp>
#include <ubu/platform/cuda/device_executor.hpp>
#include <ubu/platform/cuda/managed_allocator.hpp>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string_view>
#include <vector>


template<std::random_access_iterator I, std::random_access_iterator O, binary_invocable<std::iter_value_t<I>> F>
  requires plain_old_reducible<I,O,F>
ubu::cuda::event reduce_after_at(ubu::cuda::device_executor ex, ubu::cuda::device_allocator<int> alloc, const ubu::cuda::event& before, I first, int n, O result, F op)
{
  using namespace ubu;

  constexpr int block_size = 32*6;

  int num_multiprocessors = 0;
  cudaDeviceGetAttribute(&num_multiprocessors, cudaDevAttrMultiProcessorCount, ex.device());

  // 695 was determined empirically
  int max_num_ctas = num_multiprocessors * 695;

  // 11 was revealed in a dream
  int min_tile_size = block_size * 11;

  using namespace std::views;
  auto tiles = tile_evenly(counted(first,n), max_num_ctas, min_tile_size);

  int num_blocks = tiles.size();

  if(num_blocks > 1)
  {
    // allocate storage for each tile's reduction
    using partial_sum_type = std::invoke_result_t<F,std::iter_value_t<I>,std::iter_value_t<I>>;

    auto [allocation_ready, partial_results] = allocate_after<partial_sum_type>(alloc, before, num_blocks);

    // reduce each tile into a partial result
    auto first_phase = ex.bulk_execute_after(allocation_ready, ubu::int2(block_size,num_blocks), [=](ubu::int2 idx)
    {
      auto [thread,block] = idx;
      reduce_tiles_kernel<block_size>(block, thread, tiles, partial_results, op);
    });

    // finish up in a second phase by reducing the partial results
    auto single_tile_of_partial_results = tile(counted(partial_results, num_blocks), num_blocks);
    auto second_phase = ex.bulk_execute_after(first_phase, ubu::int2(512,1), [=](ubu::int2 idx)
    {
      auto [thread,block] = idx;
      reduce_tiles_kernel<512>(0, thread, single_tile_of_partial_results, result, op);
    });

    // deallocate storage
    return deallocate_after(alloc, second_phase, partial_results, num_blocks);
  }

  // the input is small enough that it only requires a single phase 
  return ex.bulk_execute_after(before, ubu::int2(512,1), [=](ubu::int2 idx)
  {
    auto [thread,block] = idx;
    reduce_tiles_kernel<512>(0, thread, tiles, result, op);
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
  ubu::cuda::device_executor ex;
  ubu::cuda::device_allocator<int> alloc;
  ubu::cuda::event before = ubu::initial_happening(ex);

  for(int size = 1000; size < max_size; size += size / 100)
  {
    reduce_after_at(ex, alloc, before, input.data(), size, result.data(), std::plus{}).wait();

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
  ubu::cuda::device_executor ex;
  ubu::cuda::device_allocator<int> alloc;
  ubu::cuda::event before = ubu::initial_happening(ex);

  // warmup
  reduce_after_at(ex, alloc, before, input.data(), size, result.data(), std::plus{});

  return measure_bandwidth_of_invocation_in_gigabytes_per_second(num_trials, sizeof(int) * input.size(), [&]
  {
    reduce_after_at(ex, alloc, before, input.data(), size, result.data(), std::plus{});
  });
}


int main(int argc, char** argv)
{
  if(argc == 2)
  {
    std::string_view arg(argv[1]);
    if(arg != "quick")
    {
      std::cerr << "Unrecognized argument \"" << arg << "\"" << std::endl;
      return -1;
    }

    test_correctness(1 << 16);
    std::cout << "OK" << std::endl;
    return 0;
  }

  std::cout << "Testing correctness... " << std::flush;
  test_correctness(23456789);
  std::cout << "Done." << std::endl;

  std::cout << "Testing performance... " << std::flush;
  double bandwidth = test_performance(1 << 30, 1000);
  std::cout << "Done." << std::endl;

  std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

  return 0; 
}

