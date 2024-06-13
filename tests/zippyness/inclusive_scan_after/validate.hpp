#pragma once

#include <compare>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <format>
#include <map>
#include <stdexcept>
#include <string_view>
#include <ubu/ubu.hpp>
#include <vector>

template<class T>
constexpr std::vector<T> to_host(const std::vector<T, ubu::cuda::managed_allocator<T>>& vec)
{
  return {vec.begin(), vec.end()};
}

template<ubu::tensor_like T>
  requires std::is_trivially_copyable_v<T>
constexpr T to_host(T tensor)
{
  return tensor;
}

template<class T>
std::vector<std::size_t> mismatches(const std::vector<T>& expected, const std::vector<T>& test)
{
  assert(expected.size() == test.size());

  std::vector<std::size_t> result;

  for(std::size_t i = 0; i < expected.size(); ++i)
  {
    if(i < test.size())
    {
      if(expected[i] != test[i])
      {
        result.push_back(i);
      }
    }
    else
    {
      result.push_back(i);
    }
  }

  return result;
}

template<class T>
void validate_result(const std::vector<T>& expected, const std::vector<T>& input, const std::vector<T>& result, const std::string& test_name = "test")
{
  if(expected.size() != result.size())
  {
    std::string what = fmt::format("{} failed\n", test_name);
    what += fmt::format("Unexpected result size: expected {}, got {}\n", expected.size(), result.size());
    throw std::runtime_error(what);
  }

  if(expected != result)
  {
    std::string what = fmt::format("{} failed\n", test_name);

    if(expected.size() < 50)
    {
      what += fmt::format("input:    {}\n", input);
      what += fmt::format("expected: {}\n", expected);
      what += fmt::format("result:   {}\n", result);
    }

    int i = 0;
    for(auto idx : mismatches(expected, result))
    {
      what += fmt::format("Unexpected result[{}]: expected {}, got {}\n", idx, expected[idx], result[idx]);

      if(++i == 10)
      {
        what += "...\n";
        break;
      }
    }

    throw std::runtime_error(what);
  }
}

template<class T>
void validate_result(const T& expected, const T& result, const std::string& test_name = "test")
{
  if(expected != result)
  {
    std::string what = fmt::format("{} failed\n", test_name);
    what += fmt::format("expected: {}\n", expected);
    what += fmt::format("result:   {}\n", result);

    throw std::runtime_error(what);
  }
}

template<std::invocable R, std::invocable F>
void validate(R reference, F test, const std::string& test_name = "test")
{
  // compute the expected result from the reference
  auto expected = reference();

  // compute the result of the function
  auto result = test();

  // check the result
  validate_result(expected, result, test_name);
}

std::vector<std::size_t> test_sizes(std::size_t max_size)
{
  std::vector<std::size_t> result = {0, 1, 2};

  int max_num_warps_per_block = 32;

  for(int num_warps = 1; num_warps <= max_num_warps_per_block; ++num_warps)
  {
    int block_size = ubu::cuda::warp_size * num_warps;

    result.push_back(block_size-1);
    result.push_back(block_size);
    result.push_back(block_size+1);
  }

  for(std::size_t size = 10000; size < max_size; size += size / 10)
  {
    result.push_back(size);
  }

  result.push_back(max_size-1);
  result.push_back(max_size);

  std::sort(result.begin(), result.end());

  return result;
}

std::string_view device_name(int device = 0)
{
  cudaDeviceProp prop;
  if(cudaError_t error = cudaGetDeviceProperties(&prop, device))
  {
    throw std::runtime_error(std::format("device_name: CUDA error after cudaGetDeviceProperties: {}", cudaGetErrorString(error)));
  }

  return prop.name;
}

int multiprocessor_count(int device = 0)
{
  cudaDeviceProp deviceProp;
  if(cudaError_t error = cudaGetDeviceProperties(&deviceProp, device))
  {
    throw std::runtime_error(std::format("multiprocessor_count: CUDA error after cudaGetDeviceProperties: {}", cudaGetErrorString(error)));
  }

  return deviceProp.multiProcessorCount;
}

template<class T>
std::size_t choose_large_problem_size_using_heuristic(int number_of_arrays_used_by_algorithm, int device = 0)
{
  // we haven't got all day
  std::size_t largest_size_we_are_willing_to_wait_for = 1ul << 32;

  // we generally won't be able to allocate the entire GPU memory because the system is already using some of GPU memory for other stuff
  std::size_t largest_array_size_we_can_accomodate = ubu::cuda::device_allocator<T>().max_size() / (number_of_arrays_used_by_algorithm + 1);

  return std::min(largest_size_we_are_willing_to_wait_for, largest_array_size_we_can_accomodate);
}

double theoretical_peak_bandwidth_in_gigabytes_per_second(int device = 0)
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  double memory_clock_mhz = static_cast<double>(prop.memoryClockRate) / 1000.0;
  double memory_bus_width_bits = static_cast<double>(prop.memoryBusWidth);

  return (memory_clock_mhz * memory_bus_width_bits * 2 / 8.0) / 1024.0;
}


// performance_expectations_t is a map from device name to an interval in [0,1], whose units are percent of theoretical peak bandwidth
using performance_expectations_t = std::map<std::string_view,std::pair<double,double>>;

std::pair<double,double> expected_performance(const performance_expectations_t& expectations, int device = 0)
{
  // if we don't recognize the GPU, we'll return the unit interval
  std::pair result(0., 1.);

  if(auto name = device_name(device); expectations.contains(name))
  {
    result = expectations.at(name);
  }

  return result;
}

// bandwidth is given in units of GB/s
double bandwidth_to_performance(double bandwidth, int device = 0)
{
  return bandwidth / theoretical_peak_bandwidth_in_gigabytes_per_second();
}

// performance is given in units of percent of theoretical peak bandwidth
constexpr std::partial_ordering grade_performance(double performance, const performance_expectations_t& expectations, int device = 0)
{
  auto expected = expected_performance(expectations, device);

  if(performance <= expected.first)
  {
    return std::partial_ordering::less;
  }
  else if(performance >= expected.second)
  {
    return std::partial_ordering::greater;
  }

  return std::partial_ordering::equivalent;
}

// performance is given in units of percent of theoretical peak bandwidth
void report_performance(double performance, const performance_expectations_t& expectations, int device = 0)
{
  std::cout << "Bandwidth: " << performance * theoretical_peak_bandwidth_in_gigabytes_per_second(device) << " GB/s" << std::endl;
  std::cout << "Percent peak bandwidth: " << performance << "%" << std::endl;

  if(auto grade = grade_performance(performance, expectations, device); grade < 0)
  {
    double threshold = expected_performance(expectations, device).first;
    throw std::runtime_error(std::format("Regression threshold: {}%\nRegression detected.", threshold));
  }
  else if(grade > 0)
  {
    double threshold = expected_performance(expectations, device).second;
    throw std::runtime_error(std::format("Progression threshold: {}%\nProgression detected.", threshold));
  }
}


