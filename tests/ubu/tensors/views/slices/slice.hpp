#include <algorithm>
#include <cassert>
#include <iostream>
#include <ubu/tensors/concepts/tensor_like_of_rank.hpp>
#include <ubu/tensors/shapes/shape_size.hpp>
#include <ubu/tensors/views/composed_view.hpp>
#include <ubu/tensors/views/domain.hpp>
#include <ubu/tensors/views/lattice.hpp>
#include <ubu/tensors/views/layouts/identity_layout.hpp>
#include <ubu/tensors/views/layouts/strided_layout.hpp>
#include <ubu/tensors/views/slices/slice.hpp>

namespace ns = ubu;

constexpr int ceil_div(int n, int d)
{
  return (n + d - 1) / d;
}

constexpr int tile_evenly(int n, int desired_number_of_tiles, int minimum_tile_size)
{
  int tentative_tile_size = ceil_div(n, desired_number_of_tiles);
  return std::max(tentative_tile_size, minimum_tile_size);
}

template<class T>
constexpr auto partition_for_reduction(const T* ptr, std::size_t n)
{
  // hardware constants
  int warp_size = 32;
  int num_sms = 32;

  // tuning knobs
  int min_work_per_thread = 11;
  int num_warps_per_block = 6;
  int max_num_blocks_per_sm = 2;

  // configuration
  int block_size = num_warps_per_block * warp_size;
  int max_num_blocks = num_sms * max_num_blocks_per_sm;
  int min_tile_size = block_size * min_work_per_thread;

  // divide the input tiles
  int tile_size = tile_evenly(n, max_num_blocks, min_tile_size);
  int num_tiles = ceil_div(n, tile_size);
  int work_per_thread = ceil_div(tile_size, block_size);

  // we want this resulting view of the data:
  // view[(loop_idx, thread_idx, block_idx)] = ptr + loop_idx*block_size + thread_idx + block_idx*tile_size;
  ns::int3 shape(work_per_thread, block_size, num_tiles);
  ns::int3 stride(block_size, 1, tile_size);

  return ns::composed_view(ptr, ns::strided_layout(shape, stride));
}

void test1()
{
  // this number is chosen so that it will be evenly divided by the partitioning of partition_for_reduction
  std::size_t n = 32*32*11*6*2;

  std::vector<ns::int3> data(n);

  // label each element with its coordinate of the partioning
  ns::composed_view view = partition_for_reduction(data.data(), data.size());

  assert(ns::shape_size(view.shape()) == n);

  for(auto coord : ns::domain(view))
  {
    int i = view.b()[coord];
    data[i] = coord;
  }

  {
    // test that each element is labeled as expected

    for(auto coord : ns::domain(view))
    {
      if(coord != view[coord])
      {
        std::cerr << "coord: " << coord << ", view[coord]: " << view[coord] << std::endl;
      }
      assert(coord == view[coord]);
    }
  }

  {
    // slice along each outer tile of the view and test that
    // each element of the tile is labeled as expected

    for(int tile_idx = 0; tile_idx != view.shape()[2]; ++tile_idx)
    {
      auto tile = ns::slice(view, std::tuple(ns::_, ns::_, tile_idx));

      for(auto ij : ns::domain(tile))
      {
        ns::int3 ijk(ij[0], ij[1], tile_idx);
        if(tile[ij] != view[ijk])
        {
          std::cerr << "ijk: " << ijk << ", view[ijk]: " << view[ijk] << std::endl;
        }
        assert(tile[ij] == view[ijk]);
      }
    }
  }
}

void test2()
{
  using namespace ns;

  lattice tensor(ns::int3(3,4,5));

  // slice each inner dimension
  for(int i = 0; i != shape(tensor)[0]; ++i)
  {
    auto s = slice(tensor, std::tuple(i, _, _));

    for(ns::int2 jk : domain(s))
    {
      ns::int3 ijk(i, jk[0], jk[1]);

      if(s[jk] != tensor[ijk])
      {
        std::cerr << "ijk: " << ijk << ", tensor[ijk]: " << tensor[ijk] << std::endl;
      }
      assert(s[jk] == tensor[ijk]);
    }
  }

  // slice each middle dimension
  for(int j = 0; j != shape(tensor)[1]; ++j)
  {
    auto s = slice(tensor, std::tuple(_, j, _));

    for(ns::int2 ik : domain(s))
    {
      ns::int3 ijk(ik[0], j, ik[1]);

      if(s[ik] != tensor[ijk])
      {
        std::cerr << "ijk: " << ijk << ", tensor[ijk]: " << tensor[ijk] << std::endl;
      }
      assert(s[ik] == tensor[ijk]);
    }
  }

  // slice each outer dimension
  for(int k = 0; k != shape(tensor)[2]; ++k)
  {
    auto s = slice(tensor, std::tuple(_, _, k));

    for(ns::int2 ij : domain(s))
    {
      ns::int3 ijk(ij[0], ij[1], k);

      if(s[ij] != tensor[ijk])
      {
        std::cerr << "ijk: " << ijk << ", tensor[ijk]: " << tensor[ijk] << std::endl;
      }
      assert(s[ij] == tensor[ijk]);
    }
  }
}

void test_scalar()
{
  using namespace ns;
  using namespace std;

  {
    // slice a lattice

    constexpr lattice tensor(ns::int2(2,3));

    constexpr auto s = slice(tensor, ns::int2(1,2));

    static_assert(tensor_like_of_rank<decltype(s),0>);

    constexpr auto expected = ns::int2(1,2);

    static_assert(expected == s[std::tuple()]);
  }

  {
    // try something complex
    constexpr tuple shape(tuple(pair(1,2),3), tuple(4,5), tuple(6));

    constexpr lattice tensor(shape);

    // pick out the final element
    constexpr auto s = slice(tensor, tuple(tuple(pair(0,1),2), tuple(3,4), tuple(5)));

    static_assert(tensor_like_of_rank<decltype(s),0>);

    constexpr auto expected = tuple(tuple(pair(0,1),2), tuple(3,4), tuple(5));

    static_assert(expected == s[std::tuple()]);
  }

  {
    // slice an identity_layout

    constexpr int rows = 4;
    constexpr int cols = 5;

    constexpr auto matrix = identity_layout(ubu::int2(rows,cols));

    constexpr auto s = slice(matrix, ubu::int2(2,3));

    static_assert(tensor_like_of_rank<decltype(s),0>);

    constexpr auto expected = ubu::int2(2,3);

    static_assert(expected == s[std::tuple()]);
  }
}

void test_slice()
{
  test1();
  test2();
  test_scalar();
}

