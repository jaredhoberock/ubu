#pragma once

#include <chrono>
#include <cuda_runtime_api.h>

template<class Function, class... Args>
double measure_bandwidth_of_invocation_in_gigabytes_per_second(std::size_t num_trials, std::size_t num_bytes, Function&& f, Args&&... args)
{
  using namespace std::chrono;

  cudaDeviceSynchronize();

  auto start = system_clock::now();

  for(std::size_t i = 0; i < num_trials; ++i)
  {
    std::forward<Function>(f)(std::forward<Args>(args)...);
  }

  cudaDeviceSynchronize();

  auto elapsed = system_clock::now() - start;

  auto usecs = duration_cast<microseconds>(elapsed).count();

  double total_seconds = double(usecs) / 1000000;

  double mean_seconds = total_seconds / num_trials;

  double bytes_per_second = double(num_bytes) / mean_seconds;

  double gigabytes_per_second = bytes_per_second / 1000000000;

  return gigabytes_per_second;
}

