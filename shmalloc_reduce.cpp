// ~/Desktop/circle/circle -std=c++20 -I. --cuda-path=/usr/local/cuda -sm_60 --verbose shmalloc_reduce.cpp -o a.out

#include "measure_bandwidth_of_invocation.hpp"
#include "shmalloc_reduce_kernel.hpp"
#include "tile.hpp"
#include <ubu/causality/first_cause.hpp>
#include <ubu/execution/executor/bulk_execute_after.hpp>
#include <ubu/platform/cuda/kernel_executor.hpp>
#include <ubu/platform/cuda/managed_allocator.hpp>
#include <functional>
#include <numeric>
#include <vector>
#include <iostream>
#include <random>


template<std::random_access_iterator I, std::random_access_iterator O1, std::random_access_iterator O2, std::invocable<std::iter_value_t<I>,std::iter_value_t<I>> F>
  requires plain_old_reducible<I,O1,F> and plain_old_reducible<I,O2,F>
void my_reduce(ubu::cuda::kernel_executor ex, I first, int n, O1 result, O2 partial_results, F op)
{
  using T = std::iter_value_t<I>;

  constexpr int block_size = 32*6;

  int num_multiprocessors = 0;
  cudaDeviceGetAttribute(&num_multiprocessors, cudaDevAttrMultiProcessorCount, ex.device());

  // 695 was determined empirically
  int max_num_ctas = num_multiprocessors * 695;

  // 11 was revealed in a dream
  int min_tile_size = block_size * 11;

  using namespace std::views;
  auto tiles = tile_evenly(counted(first,n), max_num_ctas, min_tile_size);

  ubu::cuda::event before = ubu::first_cause(ex);

  if(tiles.size() > 1)
  {
    auto first_phase = ubu::bulk_execute_after(ex, before, {tiles.size(),block_size}, [=](ubu::int2 idx)
    {
      shmalloc_reduce_tiles_kernel<block_size>(idx.x, idx.y, tiles, partial_results, op);
    });

    // finish up in a second phase by reducing a single tile with a larger block

    constexpr int block_size = 512;
    auto single_tile_of_partial_results = tile(counted(partial_results, tiles.size()), tiles.size());
    ubu::bulk_execute_after(ex, first_phase, {1,block_size}, [=](ubu::int2 idx)
    {
      shmalloc_reduce_tiles_kernel<block_size>(idx.x, idx.y, single_tile_of_partial_results, result, op);
    });
  }
  else
  {
    // the input is small enough that it only requires a single phase 
    
    constexpr int block_size = 512;
    ubu::bulk_execute_after(ex, before, {1,block_size}, [=](ubu::int2 idx)
    {
      shmalloc_reduce_tiles_kernel<block_size>(idx.x, idx.y, tiles, result, op);
    });
  }
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

  for(int size = 1000; size < max_size; size += size / 100)
  {
    my_reduce(ex, input.data(), size, result.data(), temporary.data(), std::plus{});
    cudaStreamSynchronize(0);

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
  device_vector<int> temporary(size);
  ubu::cuda::kernel_executor ex;

  // warmup
  my_reduce(ex, input.data(), size, result.data(), temporary.data(), std::plus{});

  return measure_bandwidth_of_invocation_in_gigabytes_per_second(num_trials, sizeof(int) * input.size(), [&]
  {
    my_reduce(ex, input.data(), size, result.data(), temporary.data(), std::plus{});
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

