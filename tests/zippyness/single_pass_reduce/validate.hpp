#pragma once

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <stdexcept>
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

