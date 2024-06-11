#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/views/lattice.hpp>

#undef NDEBUG
#include <cassert>

void test_reshape()
{
  namespace ns = ubu;

  {
    ns::lattice<int> l{13};

    int expected = 7;
    l.reshape(expected);

    assert(expected == l.shape());
  }

  {
    ns::lattice<ns::int2> l{{13,7}};

    ns::int2 expected{1,2};
    l.reshape(expected);

    assert(expected == l.shape());
  }

  {
    ns::lattice<ns::int3> l{{13,7,42}};

    ns::int3 expected{1,2,3};
    l.reshape(expected);

    assert(expected == l.shape());
  }
}

