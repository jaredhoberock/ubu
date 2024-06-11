#include <cassert>
#include <optional>
#include <random>
#include <span>
#include <ubu/ubu.hpp>
#include <vector>

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

device_vector<int> choose_unique_numbers(int n, int max)
{
  using namespace std;

  device_vector<int> result(max);
  iota(result.begin(), result.end(), 0);

  // shuffle randomly
  default_random_engine g;
  shuffle(result.begin(), result.end(), g);

  // return n numbers
  result.resize(n);

  return result;
}

void test_correctness(int block_size, int num_valid)
{
#if defined(__CUDACC__)
  using namespace ubu;

  device_vector<std::optional<int>> input(block_size);
  std::iota(input.begin(), input.end(), 0);
  std::span input_view(input.data(), input.size());

  // randomly create invalid input
  for(auto i : choose_unique_numbers(block_size - num_valid, block_size))
  {
    input[i] = std::nullopt;
  }

  // randomly create source indices
  device_vector<int> sources = choose_unique_numbers(block_size, block_size);
  std::span sources_view(sources.data(), sources.size());

  device_vector<std::optional<int>> result(block_size);
  std::span result_view(result.data(), result.size());

  int num_warps = block_size / cuda::warp_size;
  int local_memory_size = (sizeof(int) * block_size) + (sizeof(int) * num_warps);

  bulk_execute_with_workspace(cuda::device_executor(),
                              cuda::device_allocator<std::byte>(),
                              ubu::int2(block_size, 1),
                              ubu::int2(local_memory_size, 0),
                              [=](ubu::int2 coord, auto ws)
  {
    basic_cooperator block(coord.x, block_size, get_local_workspace(ws));

    std::optional value = input_view[coord.x];
    int source = sources_view[coord.x];
    result_view[coord.x] = coop_gather(block, value, source);
  });

  std::vector<std::optional<int>> expected(block_size);
  for(int i = 0; i < block_size; ++i)
  {
    expected[i] = input[sources[i]];
  }

  if(not std::equal(expected.begin(), expected.end(), result.begin()))
  {
    std::cerr << "test(" << block_size << ", " << num_valid << ") failed" << std::endl;
    assert(false);
  }
#endif
}

void test_coop_gather()
{
  for(int num_warps = 1; num_warps <= 32; ++num_warps)
  {
    int block_size = ubu::cuda::warp_size * num_warps;

    // checking each possible value of num_valid takes too long (~5s)
    //for(int num_valid = 0; num_valid < block_size; ++num_valid)
    for(int num_valid = 1; num_valid < block_size; num_valid *= 2)
    {
      test_correctness(block_size, num_valid);
    }
  }
}

