#include <ubu/tensor/coordinate/point.hpp>
#include <ubu/tensor/lattice.hpp>

#undef NDEBUG
#include <cassert>

void test_copy_constructor()
{
  namespace ns = ubu;

  {
    ns::lattice<int> expected{13};
    ns::lattice<int> result{expected};

    assert(expected == result);
  }

  {
    ns::lattice<ns::int2> expected{{13,7}};
    ns::lattice<ns::int2> result{expected};

    assert(expected == result);
  }

  {
    ns::lattice<ns::int3> expected{{13,7,42}};
    ns::lattice<ns::int3> result{expected};

    assert(expected == result);
  }
}

