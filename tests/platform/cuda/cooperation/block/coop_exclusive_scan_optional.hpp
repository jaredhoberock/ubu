#include <cassert>
#include <format>
#include <optional>
#include <random>
#include <span>
#include <ubu/ubu.hpp>
#include <vector>

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

void test(int block_size, int num_holes, bool with_init)
{
#if defined(__CUDACC__)
  using namespace ubu;

  device_vector<std::optional<int>> input(block_size, 1);
  std::generate(input.begin(), input.end(), std::default_random_engine());
  std::span input_view(input);

  // randomly create missing input
  for(auto i : choose_unique_numbers(num_holes))
  {
    input[i] = std::nullopt;
  }

  device_vector<std::optional<int>> result(block_size);
  std::span result_view(result);

  int num_warps = block_size / cuda::warp_size;
  int block_workspace_size = sizeof(std::optional<int>) * num_warps;

  std::optional init = with_init ? std::nullopt : std::optional(13);
  auto op = std::plus();

  bulk_execute_with_workspace(cuda::device_executor(),
                              cuda::device_allocator<std::byte>(),
                              ubu::int2(block_size, 1),
                              ubu::int2(block_workspace_size, 0),
                              [=](ubu::int2 coord, auto ws)
  {
    basic_cooperator block(coord.x, block_size, get_local_workspace(ws));

    std::optional value = input_view[coord.x];

    std::optional result = coop_exclusive_scan(block, init, value, op);

    result_view[coord.x] = result;
  });

  std::vector<std::optional<int>> expected(block_size);
  std::optional<int> cumulative_sum = init;
  for(int i = 0; i < input.size(); ++i)
  {
    expected[i] = cumulative_sum;

    if(input[i])
    {
      if(cumulative_sum)
      {
        cumulative_sum = op(*cumulative_sum, *input[i]);
      }
      else
      {
        cumulative_sum = input[i];
      }
    }
  }

  if(not std::equal(expected.begin(), expected.end(), result.begin()))
  {
    std::cerr << std::format("test({}, {}, {}) failed\n", block_size, num_holes, with_init) << std::endl;
    assert(false);
  }
#endif
}

void test_coop_exclusive_scan_optional()
{
  for(int num_warps = 1; num_warps <= 32; ++num_warps)
  {
    int block_size = num_warps * ubu::cuda::warp_size;

    // testing each possible value of num_holes takes too long (~5s)
    //for(int num_holes = 0; num_holes <= block_size; ++num_holes)
    for(int num_holes = 1; num_holes <= block_size; num_holes *= 2)
    {
      test(block_size, num_holes, false);
      test(block_size, num_holes, true);
    }
  }
}

