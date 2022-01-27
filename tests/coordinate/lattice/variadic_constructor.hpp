#include <aspera/coordinate/point.hpp>
#include <aspera/coordinate/lattice.hpp>

#undef NDEBUG
#include <cassert>

void test_variadic_constructor()
{
  namespace ns = aspera;

  {
    int expected_origin{0};
    int expected_shape{13};

    ns::lattice<int> l(expected_shape);

    assert(expected_origin == l.origin());
    assert(expected_shape == l.shape());
  }

  {
    ns::int2 expected_origin{0,0};
    ns::int2 expected_shape{13,7};

    ns::lattice<ns::int2> l(expected_shape.x, expected_shape.y);

    assert(expected_origin == l.origin());
    assert(expected_shape == l.shape());
  }

  {
    ns::int3 expected_origin{0,0,0};
    ns::int3 expected_shape{13,7,42};

    ns::lattice<ns::int3> l(expected_shape.x, expected_shape.y, expected_shape.z);

    assert(expected_origin == l.origin());
    assert(expected_shape == l.shape());
  }
}

