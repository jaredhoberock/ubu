#include <algorithm>
#include <ubu/tensor/coordinate/comparisons.hpp>
#include <ubu/tensor/coordinate/point.hpp>
#include <ubu/tensor/views/lattice.hpp>

#undef NDEBUG
#include <cassert>

void test_lex_begin_end()
{
  namespace ns = ubu;

  {
    int origin = 13;
    int shape = 7;

    ns::lattice<int> l(origin, shape);

    int linear_idx = 0;
    for(auto i = l.lex_begin(); i != l.lex_end(); ++i, ++linear_idx)
    {
      auto result = *i;
      auto expected = l[linear_idx];
      assert(expected == result);
    }

    assert(std::is_sorted(l.lex_begin(), l.lex_end(), ns::lex_less));
  }

  {
    ns::int2 origin = {13,7};
    ns::int2 shape = {3,2};

    ns::lattice<ns::int2> l(origin, shape);

    int linear_idx = 0;
    for(auto i = l.lex_begin(); i != l.lex_end(); ++i, ++linear_idx)
    {
      auto result = *i;
      auto expected = ns::lexicographical_lift(linear_idx, shape) + origin;
      if(expected != result)
      {
        std::cout << "linear_idx: " << linear_idx << std::endl;
        std::cout << "expected: " << expected << std::endl;
        std::cout << "result: " << result << std::endl;
      }
      assert(expected == result);
    }

    assert(std::is_sorted(l.lex_begin(), l.lex_end(), ns::lex_less));
  }

  {
    ns::int3 origin = {13,7,42};
    ns::int3 shape = {3,2,1};

    ns::lattice<ns::int3> l(origin, shape);

    int linear_idx = 0;
    for(auto i = l.lex_begin(); i != l.lex_end(); ++i, ++linear_idx)
    {
      auto result = *i;
      auto expected = ns::lexicographical_lift(linear_idx, shape) + origin;
      assert(expected == result);
    }

    assert(std::is_sorted(l.lex_begin(), l.lex_end(), ns::lex_less));
  }
}

