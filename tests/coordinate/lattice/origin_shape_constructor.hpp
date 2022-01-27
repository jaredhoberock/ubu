#include <aspera/coordinate/point.hpp>
#include <aspera/coordinate/lattice.hpp>

#undef NDEBUG
#include <cassert>

void test_origin_shape_constructor()
{
  namespace ns = aspera;

  {
    int expected_origin = 13;
    int expected_shape = 7;

    ns::lattice<int> l(expected_origin, expected_shape);

    assert(expected_origin == l.origin());
    assert(expected_shape == l.shape());
  }

  {
    ns::int2 expected_origin = {13,7};
    ns::int2 expected_shape = {1,2};

    ns::lattice<ns::int2> l(expected_origin, expected_shape);

    assert(expected_origin == l.origin());
    assert(expected_shape == l.shape());
  }

  {
    ns::int3 expected_origin = {13,7,42};
    ns::int3 expected_shape = {1,2,3};

    ns::lattice<ns::int3> l(expected_origin, expected_shape);

    assert(expected_origin == l.origin());
    assert(expected_shape == l.shape());
  }
}

