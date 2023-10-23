#include <algorithm>
#include <cassert>
#include <iostream>
#include <ubu/grid/domain.hpp>
#include <ubu/grid/lattice.hpp>
#include <ubu/grid/layout/strided_layout.hpp>
#include <ubu/grid/shape/shape_size.hpp>
#include <ubu/grid/slice/slice.hpp>
#include <ubu/grid/view.hpp>

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

  return ns::view(ptr, ns::strided_layout(shape, stride));
}

void test1()
{
  // this number is chosen so that it will be evenly divided by the partitioning of partition_for_reduction
  std::size_t n = 32*32*11*6*2;

  std::vector<ns::int3> data(n);

  // label each element with its coordinate of the partioning
  ns::view view = partition_for_reduction(data.data(), data.size());

  assert(ns::shape_size(view.shape()) == n);

  for(auto coord : ns::domain(view))
  {
    int i = view.layout()[coord];
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

  lattice grid(ns::int3(3,4,5));

  // slice each inner dimension
  for(int i = 0; i != shape(grid)[0]; ++i)
  {
    auto s = slice(grid, std::tuple(i, _, _));

    for(ns::int2 jk : domain(s))
    {
      ns::int3 ijk(i, jk[0], jk[1]);

      if(s[jk] != grid[ijk])
      {
        std::cerr << "ijk: " << ijk << ", grid[ijk]: " << grid[ijk] << std::endl;
      }
      assert(s[jk] == grid[ijk]);
    }
  }

  // slice each middle dimension
  for(int j = 0; j != shape(grid)[1]; ++j)
  {
    auto s = slice(grid, std::tuple(_, j, _));

    for(ns::int2 ik : domain(s))
    {
      ns::int3 ijk(ik[0], j, ik[1]);

      if(s[ik] != grid[ijk])
      {
        std::cerr << "ijk: " << ijk << ", grid[ijk]: " << grid[ijk] << std::endl;
      }
      assert(s[ik] == grid[ijk]);
    }
  }

  // slice each outer dimension
  for(int k = 0; k != shape(grid)[2]; ++k)
  {
    auto s = slice(grid, std::tuple(_, _, k));

    for(ns::int2 ij : domain(s))
    {
      ns::int3 ijk(ij[0], ij[1], k);

      if(s[ij] != grid[ijk])
      {
        std::cerr << "ijk: " << ijk << ", grid[ijk]: " << grid[ijk] << std::endl;
      }
      assert(s[ij] == grid[ijk]);
    }
  }
}

void test_slice()
{
  test1();
  test2();
}

