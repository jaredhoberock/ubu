#include <ubu/tensor/coordinate/point.hpp>
#include <ubu/tensor/views/lattice.hpp>

#undef NDEBUG
#include <cassert>

void test_bracket_operator()
{
  namespace ns = ubu;

  {
    int origin = 13;
    int shape = 7;

    ns::lattice<int> l(origin, shape);

    for(int idx = 0; idx != l.shape(); ++idx)
    {
      assert(origin + idx == l[idx]);
    }
  }

  {
    ns::int2 origin = {13,7};
    ns::int2 shape = {3,2};

    ns::lattice<ns::int2> l(origin, shape);

    for(int i = 0; i != l.shape().x; ++i)
    {
      for(int j = 0; j != l.shape().y; ++j)
      {
        ns::int2 idx{i,j};
        assert(origin + idx == l[idx]);
      }
    }
  }

  {
    ns::int3 origin = {13,7,42};
    ns::int3 shape = {1,2,3};

    ns::lattice<ns::int3> l(origin, shape);

    for(int i = 0; i != l.shape().x; ++i)
    {
      for(int j = 0; j != l.shape().y; ++j)
      {
        for(int k = 0; k != l.shape().z; ++k)
        {
          ns::int3 idx{i,j,k};
          assert(origin + idx == l[idx]);
        }
      }
    }
  }
}

