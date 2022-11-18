#include <ubu/coordinate/lexicographic_lift.hpp>
#include <ubu/coordinate/point.hpp>

#undef NDEBUG
#include <cassert>


void test_lexicographic_lift()
{
  namespace ns = ubu;

  {
    // 1D
    int shape{13};

    std::size_t index = 0;
    for(int i = 0; i < shape; ++i, ++index)
    {
      int expected = i;
      int result = ns::lexicographic_lift(index, shape);

      assert(expected == result);
    }
  }

  {
    // 2D
    ns::int2 shape{13,7};

    std::size_t index = 0;
    for(int i = 0; i < shape[0]; ++i)
    {
      for(int j = 0; j < shape[1]; ++j, ++index)
      {
        ns::int2 expected{i,j};
        ns::int2 result = ns::lexicographic_lift(index, shape);

        assert(expected == result);
      }
    }
  }

  {
    // 3D
    ns::int3 shape{13,7,42};

    std::size_t index = 0;
    for(int i = 0; i < shape[0]; ++i)
    {
      for(int j = 0; j < shape[1]; ++j)
      {
        for(int k = 0; k < shape[2]; ++k, ++index)
        {
          ns::int3 expected{i,j,k};
          ns::int3 result = ns::lexicographic_lift(index, shape);

          assert(expected == result);
        }
      }
    }
  }

  {
    // 2D x 3D
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
              std::pair<ns::int2,ns::int3> expected{{i,j}, {x,y,z}};
              std::pair<ns::int2,ns::int3> result = ns::lexicographic_lift(index, shape);

              assert(expected == result);
            }
          }
        }
      }
    }
  }
}

