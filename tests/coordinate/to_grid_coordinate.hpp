#include <aspera/coordinate/detail/compact_column_major_stride.hpp>
#include <aspera/coordinate/detail/compact_row_major_stride.hpp>
#include <aspera/coordinate/point.hpp>
#include <aspera/coordinate/to_grid_coordinate.hpp>
#include <cassert>


void test_to_grid_coordinate()
{
  namespace ns = aspera;

  {
    // 1D, column major
    int shape{13};

    std::size_t index = 0;
    for(int coord = 0; coord < shape; ++coord, ++index)
    {
      int expected = coord;

      int result = ns::to_grid_coordinate<int>(index, shape, ns::detail::compact_column_major_stride(shape));

      assert(expected == result);
    }
  }

  {
    // 1D, row major
    int shape{13};

    std::size_t index = 0;
    for(int coord = 0; coord < shape; ++coord, ++index)
    {
      int expected = coord;

      int result = ns::to_grid_coordinate<int>(index, shape, ns::detail::compact_row_major_stride(shape));

      assert(expected == result);
    }
  }

  {
    // 2D, column major
    ns::int2 shape{13,7};

    std::size_t index = 0;
    for(int j = 0; j < shape[1]; ++j)
    {
      for(int i = 0; i < shape[0]; ++i, ++index)
      {
        ns::int2 expected{i,j};
        ns::int2 result = ns::to_grid_coordinate<ns::int2>(index, shape, ns::detail::compact_column_major_stride(shape));

        assert(expected == result);
      }
    }
  }

  {
    // 2D, row major
    ns::int2 shape{13,7};

    std::size_t index = 0;
    for(int i = 0; i < shape[0]; ++i)
    {
      for(int j = 0; j < shape[1]; ++j, ++index)
      {
        ns::int2 expected{i,j};
        ns::int2 result = ns::to_grid_coordinate<ns::int2>(index, shape, ns::detail::compact_row_major_stride(shape));

        assert(expected == result);
      }
    }
  }

  {
    // 3D, column major
    ns::int3 shape{13,7,42};

    std::size_t index = 0;
    for(int k = 0; k < shape[2]; ++k)
    {
      for(int j = 0; j < shape[1]; ++j)
      {
        for(int i = 0; i < shape[0]; ++i, ++index)
        {
          ns::int3 expected{i,j,k};
          ns::int3 result = ns::to_grid_coordinate<ns::int3>(index, shape, ns::detail::compact_column_major_stride(shape));

          assert(expected == result);
        }
      }
    }
  }

  {
    // 3D, row major
    ns::int3 shape{13,7,42};

    std::size_t index = 0;
    for(int i = 0; i < shape[0]; ++i)
    {
      for(int j = 0; j < shape[1]; ++j)
      {
        for(int k = 0; k < shape[2]; ++k, ++index)
        {
          ns::int3 expected{i,j,k};
          ns::int3 result = ns::to_grid_coordinate<ns::int3>(index, shape, ns::detail::compact_row_major_stride(shape));

          assert(expected == result);
        }
      }
    }
  }

  {
    // 2D x 3D, column major
    // {{i,j}, {x,y,z}}
    std::pair<ns::int2,ns::int3> shape{{13,7}, {42,11,5}};

    std::size_t index = 0;
    for(int z = 0; z < shape.second[2]; ++z)
    {
      for(int y = 0; y < shape.second[1]; ++y)
      {
        for(int x = 0; x < shape.second[0]; ++x)
        {
          for(int j = 0; j < shape.first[1]; ++j)
          {
            for(int i = 0; i < shape.first[0]; ++i, ++index)
            {
              using coord_type = std::pair<ns::int2, ns::int3>;

              coord_type expected{{i,j}, {x,y,z}};
              coord_type result = ns::to_grid_coordinate<coord_type>(index, shape, ns::detail::compact_column_major_stride(shape));

              assert(expected == result);
            }
          }
        }
      }
    }
  }

  {
    // 2D x 3D, row major
    // {{i,j}, {x,y,z}}
    std::pair<ns::int2,ns::int3> shape{{13,7}, {42,11,5}};

    std::size_t index = 0;
    for(int i = 0; i < shape.first[0]; ++i)
    {
      for(int j = 0; j < shape.first[1]; ++j)
      {
        for(int x = 0; x < shape.second[0]; ++x)
        {
          for(int y = 0; y < shape.second[1]; ++y)
          {
            for(int z = 0; z < shape.second[2]; ++z, ++index)
            {
              using coord_type = std::pair<ns::int2, ns::int3>;

              coord_type expected{{i,j}, {x,y,z}};
              coord_type result = ns::to_grid_coordinate<coord_type>(index, shape, ns::detail::compact_row_major_stride(shape));

              assert(expected == result);
            }
          }
        }
      }
    }
  }
}

