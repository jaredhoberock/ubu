#include <cassert>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <ubu/ubu.hpp>
#include <vector>

template<class V1, class V2>
std::vector<std::size_t> mismatches(const V1& expected, const V2& test)
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
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

std::vector<int> choose_unique_numbers(int n)
{
  using namespace std;

  vector<int> result(32);
  iota(result.begin(), result.end(), 0);

  // shuffle randomly
  default_random_engine g;
  std::shuffle(result.begin(), result.end(), g);

  // return n numbers
  result.resize(n);

  return result;
}

void test_warp_like_inclusive_scan(int num_holes)
{
#if defined(__CUDACC__)
  using namespace ubu;

  device_vector<std::optional<int>> input(ubu::cuda::warp_size, 1);
  std::span input_view(input.data(), input.size());

  // randomly create missing input
  for(auto i : choose_unique_numbers(num_holes))
  {
    input[i] = std::nullopt;
  }

  device_vector<std::optional<int>> result(ubu::cuda::warp_size);
  std::span result_view(result.data(), result.size());

  bulk_execute(cuda::device_executor(), ubu::int2(cuda::warp_size, 1), [=](ubu::int2 coord)
  {
    basic_cooperator warp(coord.x, cuda::warp_size, cuda::warp_workspace{});

    std::optional value = input_view[coord.x];

    std::optional result = coop_inclusive_scan(warp, value, std::plus{});

    result_view[coord.x] = result;
  });

  std::vector<std::optional<int>> expected(ubu::cuda::warp_size);
  std::optional<int> cumulative_sum;
  for(int i = 0; i != input.size(); ++i)
  {
    if(input[i])
    {
      if(cumulative_sum)
      {
        cumulative_sum = *cumulative_sum + *input[i];
      }
      else
      {
        cumulative_sum = input[i];
      }
    }

    expected[i] = cumulative_sum;
  }

  if(not std::equal(expected.begin(), expected.end(), result.begin()))
  {
    std::cerr << "test(" << num_holes << ") failed" << std::endl;
    assert(false);
  }
#endif
}

void test_coop_inclusive_scan()
{
  for(int num_holes = 0; num_holes < ubu::cuda::warp_size; ++num_holes)
  {
    test_warp_like_inclusive_scan(num_holes);
  }
}

