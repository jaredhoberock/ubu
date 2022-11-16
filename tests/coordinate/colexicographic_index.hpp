#include <ubu/coordinate/colexicographic_index.hpp>
#include <ubu/coordinate/point.hpp>

#undef NDEBUG
#include <cassert>

#include <cstdio>

void test_colexicographic_index()
{
  namespace ns = ubu;

  {
    // 1D
    int shape{13};

    std::size_t expected = 0;
    for(int coord = 0; coord < shape; ++coord, ++expected)
    {
      std::size_t result = ns::colexicographic_index(coord, shape);

      assert(expected == result);
    }
  }

  {
    // 2D
    ns::int2 shape{13,7};

    std::size_t expected = 0;
    for(int j = 0; j < shape[1]; ++j)
    {
      for(int i = 0; i < shape[0]; ++i, ++expected)
      {
        ns::int2 coord{i,j};
        std::size_t result = ns::colexicographic_index(coord, shape);

        assert(expected == result);
      }
    }
  }

  {
    // 3D
    ns::int3 shape{13,7,42};

    std::size_t expected = 0;
    for(int k = 0; k < shape[2]; ++k)
    {
      for(int j = 0; j < shape[1]; ++j)
      {
        for(int i = 0; i < shape[0]; ++i, ++expected)
        {
          ns::int3 coord{i,j,k};
          std::size_t result = ns::colexicographic_index(coord, shape);

          assert(expected == result);
        }
      }
    }
  }

  {
    // 2D x 3D
    // {{i,j}, {x,y,z}}
    std::pair<ns::int2,ns::int3> shape{{13,7}, {42,11,5}};

    std::size_t expected = 0;
    for(int z = 0; z < shape.second[2]; ++z)
    {
      for(int y = 0; y < shape.second[1]; ++y)
      {
        for(int x = 0; x < shape.second[0]; ++x)
        {
          for(int j = 0; j < shape.first[1]; ++j)
          {
            for(int i = 0; i < shape.first[0]; ++i, ++expected)
            {
              std::pair<ns::int2, ns::int3> coord{{i,j}, {x,y,z}};
              std::size_t result = ns::colexicographic_index(coord, shape);

              assert(expected == result);
            }
          }
        }
      }
    }
  }
}

