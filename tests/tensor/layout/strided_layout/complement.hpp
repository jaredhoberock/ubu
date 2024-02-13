#include <concepts>
#include <ubu/tensor/layout/strided_layout.hpp>

#undef NDEBUG
#include <cassert>


namespace ns = ubu;


template<std::integral I>
constexpr I ceil_div(I n, I d)
{
  return (n + d - 1) / d;
}


template<ns::coordinate S, ns::stride_for<S> D, class CoSizeHi>
bool test_complement(ns::strided_layout<S,D> layout, CoSizeHi cosize_hi)
{
  using namespace ns;

  strided_layout result = layout.complement(cosize_hi);

  // post-condition on the domain size of the complement
  // XXX cute does a filter on the layout first
  if(not (shape_size(result.shape()) >= cosize_hi / shape_size(layout.shape())))
  {
    return false;
  }

  // post-condition on the codomain size of the complement
  if(not (shape_size(result.coshape()) <= ceil_div(cosize_hi, shape_size(layout.coshape())) * shape_size(layout.coshape())))
  {
    return false;
  }

  // post-condition on the codomain of the complement
  for(int i = 1; i < shape_size(result.shape()); ++i)
  {
    if(not (result[i-1] < result[i]))
    {
      return false;
    }

    for(int j = 0; j < shape_size(layout.shape()); ++j)
    {
      if(not (result[i] != layout[j]))
      {
        return false;
      }
    }
  }

  if(not (shape_size(result.shape()) <= shape_size(result.shape())))
  {
    return false;
  }

  if(not (shape_size(result.coshape()) >= cosize_hi / shape_size(layout.shape())))
  {
    return false;
  }

  return true;
}


template<ns::coordinate S, ns::stride_for<S> D>
bool test_complement(ns::strided_layout<S,D> layout)
{
  return test_complement(layout, ns::shape_size(layout.coshape()));
}


void test_complement()
{
  using namespace std;
  using namespace ns;

  //{
  //  // XXX this fails due to division by zero
  //  strided_layout layout(1,0);

  //  assert(test_complement(layout));
  //}

  {
    strided_layout layout(1,1);

    assert(test_complement(layout));
    assert(test_complement(layout, 2));
  }

  //{
  //  // XXX this fails due to division by zero
  //  strided_layout layout(4,0);
  //  test_complement(layout);
  //}

  {
    strided_layout layout(ns::int2(2,4), ns::int2(1,2));
    assert(test_complement(layout));
  }

  {
    strided_layout layout(ns::int2(2,3), ns::int2(1,2));
    assert(test_complement(layout));
  }

  {
    strided_layout layout(ns::int2(2,4), ns::int2(1,4));
    assert(test_complement(layout));
  }

  {
    strided_layout layout(ns::int3(2,4,8), ns::int3(8,1,64));
    assert(test_complement(layout));
  }

  {
    strided_layout layout{pair(pair(2,2),pair(2,2)), pair(pair(1,4),pair(8,32))};
    assert(test_complement(layout));
  }

  {
    strided_layout layout{pair(2,pair(3,4)), pair(3,pair(1,6))};
    assert(test_complement(layout));
  }

  {
    strided_layout layout{pair(4,6), pair(1,6)};
    assert(test_complement(layout));
  }

  {
    strided_layout layout{pair(4,10), pair(1,10)};
    assert(test_complement(layout));
  }

  {
    strided_layout layout(1,2);

    // XXX these fail postconditions
    assert(not test_complement(layout));
    assert(not test_complement(layout, 1));

    assert(test_complement(layout, 2));
    assert(test_complement(layout, 8));
  }
}

