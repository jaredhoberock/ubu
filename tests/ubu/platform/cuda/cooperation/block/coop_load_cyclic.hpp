#include <algorithm>
#include <numeric>
#include <ubu/ubu.hpp>
#include <vector>

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

template<int max_num_elements_per_thread>
void test_coop_load_cyclic(int num_warps_per_block, int num_elements_per_block)
{
#if defined(__CUDACC__)
  using namespace ubu;

  int block_size = num_warps_per_block * cuda::warp_size;

  device_vector<int> input(num_elements_per_block);
  std::iota(input.begin(), input.end(), 0);
  std::span input_view(input.data(), input.size());

  device_vector<int> result(input.size());
  std::span result_view(result.begin(), result.size());

  ubu::int2 shape(block_size,1);

  bulk_execute(cuda::device_executor(),
               shape,
               [=](ubu::int2 coord)
  {
    basic_cooperator block(coord.x, shape.x, cuda::block_workspace());

    inplace_vector thread_values = coop_load_cyclic<max_num_elements_per_thread>(block, input_view);

    int* result_ptr = result_view.data() + thread_values.size() * coord.x;
    for(auto x : thread_values)
    {
      *result_ptr = x;
      ++result_ptr;
    }
  });

  std::vector<int> expected(input.size());
  int* expected_ptr = expected.data();
  for(int thread_idx = 0; thread_idx < block_size; ++thread_idx)
  {
    for(int i = 0; i < max_num_elements_per_thread; ++i)
    {
      if(thread_idx + i * block_size < input.size())
      {
        *expected_ptr = thread_idx + i * block_size;
        ++expected_ptr;
      }
    }
  }

  if(not std::equal(expected.begin(), expected.end(), result.begin()))
  {
    std::cerr << "test_coop_load_cyclic(" << num_elements_per_block << ", " << num_elements_per_block << ") failed" << std::endl;
    assert(false);
  }
#endif
}

template<int max_num_elements_per_thread>
void test_coop_load_cyclic()
{
  // this number was observed empirically for sm_60
  int max_smem_capacity = 24 * ubu::cuda::warp_size * 16;

  int max_num_warps_per_block = 32;
  int max_num_elements_per_block = max_num_elements_per_thread * max_num_warps_per_block * ubu::cuda::warp_size;

  // test many possible block sizes
  for(int num_warps_per_block = 1; num_warps_per_block <= max_num_warps_per_block; num_warps_per_block *= 2)
  {
    int block_size = num_warps_per_block * ubu::cuda::warp_size;

    // test many possible elements per block
    for(int num_elements_per_block = 1; num_elements_per_block <= max_num_elements_per_block; num_elements_per_block *= 2)
    {
      if(num_elements_per_block > block_size * max_num_elements_per_thread)
      {
        // don't test input sizes that exceed our ability to load
        break;
      }

      if(max_num_elements_per_thread * block_size > max_smem_capacity)
      {
        // don't test input sizes that exceed smem capacity
        break;
      }

      test_coop_load_cyclic<max_num_elements_per_thread>(num_warps_per_block, num_elements_per_block);
    }
  }
}

void test_coop_load_cyclic()
{
  // test various values of max_num_elements_per_thread up to 16
  test_coop_load_cyclic<0>();
  test_coop_load_cyclic<1>();
  test_coop_load_cyclic<3>();
  test_coop_load_cyclic<7>();
  test_coop_load_cyclic<11>();
  test_coop_load_cyclic<15>();
  test_coop_load_cyclic<16>();
}

