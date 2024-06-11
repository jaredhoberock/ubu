#include <ubu/tensor/coordinates/point.hpp>
#include <ubu/tensor/views/lattice.hpp>

#undef NDEBUG
#include <cassert>

void test_size()
{
  namespace ns = ubu;

  {
    int origin = 13;
    int shape = 7;

    ns::lattice<int> l(origin, shape);

    assert(shape == l.size());
  }

  {
    ns::int2 origin = {13,7};
    ns::int2 shape = {1,2};

    ns::lattice<ns::int2> l(origin, shape);

    assert(shape.x * shape.y == l.size());
  }

  {
    ns::int3 origin = {13,7,42};
    ns::int3 shape = {1,2,3};

    ns::lattice<ns::int3> l(origin, shape);

    assert(shape.x * shape.y * shape.z == l.size());
  }
}

