#include <aspera/coordinate/lattice.hpp>
#include <aspera/coordinate/point.hpp>
#include <cassert>

void test_empty()
{
  namespace ns = aspera;

  {
    int origin = 13;
    int shape = 7;

    ns::lattice<int> l(origin, shape);

    assert(!l.empty());
    assert((ns::lattice<int>{}.empty()));
    assert((ns::lattice<int>{origin, {}}.empty()));
  }

  {
    ns::int2 origin = {13,7};
    ns::int2 shape = {1,2};

    ns::lattice<ns::int2> l(origin, shape);

    assert(!l.empty());
    assert((ns::lattice<ns::int2>{}.empty()));
    assert((ns::lattice<ns::int2>{origin, {}}.empty()));
  }

  {
    ns::int3 origin = {13,7,42};
    ns::int3 shape = {1,2,3};

    ns::lattice<ns::int3> l(origin, shape);

    assert(!l.empty());
    assert((ns::lattice<ns::int3>{}.empty()));
    assert((ns::lattice<ns::int3>{origin, {}}.empty()));
  }
}

