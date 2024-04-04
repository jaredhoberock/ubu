#include <algorithm>
#include <numeric>
#include <ubu/ubu.hpp>
#include <vector>

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

template<int max_num_elements_per_thread>
void test_coop_store(int num_warps_per_block, int num_elements_per_block)
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
  ubu::int2 workspace_shape(max_num_elements_per_thread * block_size * sizeof(int), 0);

  bulk_execute_with_workspace(cuda::device_executor(),
                              cuda::device_allocator<std::byte>(),
                              shape,
                              workspace_shape,
                              [=](ubu::int2 coord, auto ws)
  {
    basic_cooperator block(coord.x, shape.x, get_local_workspace(ws));

    inplace_vector thread_values = coop_load<max_num_elements_per_thread>(block, input_view);

    coop_store(block, thread_values, result_view);
  });

  std::vector<int> expected(input.begin(), input.end());
  if(not std::equal(expected.begin(), expected.end(), result.begin()))
  {
    std::cerr << "test_coop_store(" << num_elements_per_block << ", " << num_elements_per_block << ") failed" << std::endl;
    assert(false);
  }
#endif
}

template<int max_num_elements_per_thread>
void test_coop_store()
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
        // don't test input sizes that exceed our ability to store
        break;
      }

      if(max_num_elements_per_thread * block_size > max_smem_capacity)
      {
        // don't test input sizes that exceed smem capacity
        break;
      }

      test_coop_store<max_num_elements_per_thread>(num_warps_per_block, num_elements_per_block);
    }
  }
}

void test_coop_store()
{
  // test various values of max_num_elements_per_thread up to 16
  test_coop_store<0>();
  test_coop_store<1>();
  test_coop_store<3>();
  test_coop_store<7>();
  test_coop_store<11>();
  test_coop_store<16>();
}

