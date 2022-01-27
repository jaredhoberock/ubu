#include <aspera/coordinate/lattice.hpp>
#include <aspera/coordinate/point.hpp>

#undef NDEBUG
#include <cassert>

void test_contains()
{
  namespace ns = aspera;

  {
    int origin = 13;
    int shape = 7;

    ns::lattice<int> l(origin, shape);

    for(int pt = origin; pt != origin + shape; ++pt)
    {
      assert(l.contains(pt));
    }

    assert(!l.contains(origin + shape));
  }

  {
    ns::int2 origin = {13,7};
    ns::int2 shape = {3,2};

    ns::lattice<ns::int2> l(origin, shape);

    for(int i = origin.x; i != origin.x + shape.x; ++i)
    {
      for(int j = origin.y; j != origin.y + shape.y; ++j)
      {
        assert(l.contains({i,j}));
      }
    }

    assert(!l.contains(origin + shape));
  }

  {
    ns::int3 origin = {13,7,42};
    ns::int3 shape = {1,2,3};

    ns::lattice<ns::int3> l(origin, shape);

    for(int i = origin.x; i != origin.x + shape.x; ++i)
    {
      for(int j = origin.y; j != origin.y + shape.y; ++j)
      {
        for(int k = origin.z; k != origin.z + shape.z; ++k)
        {
          assert(l.contains({i,j,k}));
        }
      }
    }

    assert(!l.contains(origin + shape));
  }
}

