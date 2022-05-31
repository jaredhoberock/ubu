#include <ubu/coordinate/lattice.hpp>
#include <ubu/coordinate/point.hpp>

#undef NDEBUG
#include <cassert>

void test_begin_end()
{
  namespace ns = ubu;

  {
    int origin = 13;
    int shape = 7;

    ns::lattice<int> l(origin, shape);

    int linear_idx = 0;
    for(auto i = l.begin(); i != l.end(); ++i, ++linear_idx)
    {
      auto result = *i;
      auto expected = l[linear_idx];
      assert(expected == result);
    }
  }

  {
    ns::int2 origin = {13,7};
    ns::int2 shape = {3,2};

    ns::lattice<ns::int2> l(origin, shape);

    int linear_idx = 0;
    for(auto i = l.begin(); i != l.end(); ++i, ++linear_idx)
    {
      auto result = *i;
      auto expected = l[linear_idx];
      assert(expected == result);
    }
  }

  {
    ns::int3 origin = {13,7,42};
    ns::int3 shape = {3,2,1};

    ns::lattice<ns::int3> l(origin, shape);

    int linear_idx = 0;
    for(auto i = l.begin(); i != l.end(); ++i, ++linear_idx)
    {
      auto result = *i;
      auto expected = l[linear_idx];
      assert(expected == result);
    }
  }
}


