#include <ubu/tensor/coordinate/point.hpp>
#include <ubu/tensor/lattice.hpp>

#undef NDEBUG
#include <cassert>

void test_iterator()
{
  namespace ns = ubu;

  {
    int origin = 13;
    int shape = 7;

    ns::lattice<int> l(origin, shape);

    auto iter = l.begin();
    for(int i = 0; i != l.shape(); ++i, ++iter)
    {
      int idx = i;
      auto result = *iter;
      auto expected = origin + idx;

      assert(l.contains(result));
      assert(expected == result);
    }
  }

  {
    ns::int2 origin = {13,7};
    ns::int2 shape = {3,2};

    ns::lattice<ns::int2> l(origin, shape);

    auto iter = l.begin();
    for(int j = 0; j != l.shape()[1]; ++j)
    {
      for(int i = 0; i != l.shape()[0]; ++i, ++iter)
      {
        ns::int2 idx = {i,j};
        auto result = *iter;
        auto expected = origin + idx;

        assert(l.contains(result));
        assert(expected == result);
      }
    }
  }

  {
    ns::int3 origin = {13,7,42};
    ns::int3 shape = {3,2,1};

    ns::lattice<ns::int3> l(origin, shape);

    auto iter = l.begin();
    for(int k = 0; k != l.shape()[2]; ++k)
    {
      for(int j = 0; j != l.shape()[1]; ++j)
      {
        for(int i = 0; i != l.shape()[0]; ++i, ++iter)
        {
          ns::int3 idx{i,j,k};
          auto result = *iter;
          auto expected = origin + idx;

          assert(l.contains(result));
          assert(expected == result);
        }
      }
    }
  }
}

