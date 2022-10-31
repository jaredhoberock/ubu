#include <ubu/coordinate/to_natural_coordinate.hpp>
#include <ubu/coordinate/point.hpp>

#undef NDEBUG
#include <cassert>


void test_to_natural_coordinate()
{
  namespace ns = ubu;

  {
    // 1D
    int shape{13};

    std::size_t index = 0;
    for(int coord = 0; coord < shape; ++coord, ++index)
    {
      int expected = coord;

      int result = ns::to_natural_coordinate(index, shape);

      assert(expected == result);
    }
  }

  {
    // 2D
    ns::int2 shape{13,7};

    std::size_t index = 0;
    for(int i = 0; i < shape[1]; ++i)
    {
      for(int j = 0; j < shape[0]; ++j, ++index)
      {
        ns::int2 expected{j,i};
        ns::int2 result = ns::to_natural_coordinate(index, shape);

        assert(expected == result);
      }
    }
  }

  {
    // 3D
    ns::int3 shape{13,7,42};

    std::size_t index = 0;
    for(int i = 0; i < shape[2]; ++i)
    {
      for(int j = 0; j < shape[1]; ++j)
      {
        for(int k = 0; k < shape[0]; ++k, ++index)
        {
          ns::int3 expected{k,j,i};
          ns::int3 result = ns::to_natural_coordinate(index, shape);

          assert(expected == result);
        }
      }
    }
  }

  {
    // 3D x 2D
    // {{j,i}, {z,y,x}}
    std::pair<ns::int3,ns::int2> shape{{42,11,5}, {13,7}};

    std::size_t index = 0;
    for(int i = 0; i < shape.second[1]; ++i)
    {
      for(int j = 0; j < shape.second[0]; ++j)
      {
        for(int x = 0; x < shape.first[2]; ++x)
        {
          for(int y = 0; y < shape.first[1]; ++y)
          {
            for(int z = 0; z < shape.first[0]; ++z, ++index)
            {
              using coord_type = std::pair<ns::int3, ns::int2>;

              coord_type expected{{z,y,x}, {j,i}};
              coord_type result = ns::to_natural_coordinate(index, shape);

              assert(expected == result);
            }
          }
        }
      }
    }
  }
}

