#include <algorithm>
#include <iostream>
#include <numeric>
#include <ubu/ubu.hpp>
#include <vector>

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

template<int max_num_elements_per_thread>
void do_coop_store_columns(int block_size, ubu::span_like auto input, ubu::span_like auto result, ubu::layout_like auto matrix_layout)
{
  using namespace ubu;

  auto input_view = compose(input, matrix_layout);
  auto result_view = compose(result, matrix_layout);

  ubu::int2 shape(block_size,1);
  ubu::int2 workspace_shape(max_num_elements_per_thread * block_size * sizeof(int), 0);

  bulk_execute_with_workspace(cuda::device_executor(),
                              cuda::device_allocator<std::byte>(),
                              shape,
                              workspace_shape,
                              [=](ubu::int2 coord, auto ws)
  {
    basic_cooperator block(coord.x, shape.x, get_local_workspace(ws));

    inplace_vector thread_values = coop_load_columns(block, input_view);

    coop_store_columns(block, thread_values, result_view);
  });
}

template<int max_num_elements_per_thread>
void test_coop_store_columns(int num_warps_per_block, int num_elements_per_block)
{
#if defined(__CUDACC__)
  using namespace ubu;

  int block_size = num_warps_per_block * cuda::warp_size;

  device_vector<int> input(num_elements_per_block);
  std::iota(input.begin(), input.end(), 0);

  device_vector<int> result(input.size());

  // this is the shape of the view we will create of the input
  std::pair matrix_shape(constant<max_num_elements_per_thread>(), block_size);

  {
    // test with a column-major view of the input,
    // which is the case that is accelerated
    do_coop_store_columns<max_num_elements_per_thread>(block_size, std::span(input), std::span(result), column_major(matrix_shape));

    // check the result
    std::vector<int> expected(input.begin(), input.end());
    if(not std::equal(expected.begin(), expected.end(), result.begin()))
    {
      std::cerr << "test_coop_store_columns<" << max_num_elements_per_thread << ">(" << num_warps_per_block << ", " << num_elements_per_block << ") accelerated case failed" << std::endl;
      assert(false);
    }
  }

  result.clear();
  result.resize(input.size());

  {
    // test with a row-major view of the input,
    // which is not a case that is accelerated
    do_coop_store_columns<max_num_elements_per_thread>(block_size, std::span(input), std::span(result), row_major(matrix_shape));

    // check the result
    std::vector<int> expected(input.begin(), input.end());
    if(not std::equal(expected.begin(), expected.end(), result.begin()))
    {
      std::cerr << "test_coop_store_columns<" << max_num_elements_per_thread << ">(" << num_warps_per_block << ", " << num_elements_per_block << ") non-accelerated case failed" << std::endl;
      assert(false);
    }
  }
#endif
}

template<int max_num_elements_per_thread>
void test_coop_store_columns()
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

      test_coop_store_columns<max_num_elements_per_thread>(num_warps_per_block, num_elements_per_block);
    }
  }
}

void test_coop_store_columns()
{
  // test various values of max_num_elements_per_thread up to 16
  test_coop_store_columns<0>();
  test_coop_store_columns<1>();
  test_coop_store_columns<3>();
  test_coop_store_columns<7>();
  test_coop_store_columns<11>();
  test_coop_store_columns<16>();
}

