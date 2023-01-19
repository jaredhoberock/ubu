#include <iostream>
#include <ubu/grid.hpp>
#include <utility>

#undef NDEBUG
#include <cassert>


namespace ns = ubu;


template<class LayoutA, class LayoutB, class LayoutR>
bool test_composition(const LayoutA& layoutA,
                      const LayoutB& layoutB,
                      const LayoutR& layoutR)
{
  for(int i = 0; i < ns::grid_size(layoutR.shape()); ++i)
  {
    if(layoutR[i] != layoutA[layoutB[i]])
    {
      return false;
    }
  }

  return true;
}


template<class LayoutA, class LayoutB>
bool test_composition(const LayoutA& layoutA,
                      const LayoutB& layoutB)
{
  return test_composition(layoutA, layoutB, layoutA.compose(layoutB));
}


void test_compose()
{
  using namespace std;
  using namespace ns;

  {
    strided_layout a(1,0);
    strided_layout b(1,0);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(1,0);
    strided_layout b(1,1);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(1,1);
    strided_layout b(1,0);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(1,1);
    strided_layout b(1,1);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(4);
    strided_layout b(4);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(4,2);
    strided_layout b(4);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(4,0);
    strided_layout b(4);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(4);
    strided_layout b(4,0);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(4);
    strided_layout b(1,0);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(4);
    strided_layout b(2);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(4,2);
    strided_layout b(2);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(4);
    strided_layout b(2,2);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(4,2);
    strided_layout b(2,2);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(4,3);
    strided_layout b(12);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(12);
    strided_layout b(4,3);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(12,2);
    strided_layout b(4,3);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(2, 5);
    strided_layout b(2, 5);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(10, 2);
    strided_layout b( 5, 2);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(12);
    strided_layout b(ns::int2(4,3), ns::int2(3,1));

    assert(test_composition(a, b));
  }


  {
    strided_layout a(12, 2);
    strided_layout b(ns::int2(4,3), ns::int2(3,1));

    assert(test_composition(a, b));
  }

  {
    strided_layout a(12);
    strided_layout b(ns::int2(2,3), ns::int2(2,4));

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(4,3));
    strided_layout b(ns::int2(4,3));

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(4,3));
    strided_layout b(6);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(4,3));
    strided_layout b(6,2);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(4,3));
    strided_layout b(ns::int2(6,2), ns::int2(2,1));

    assert(test_composition(a, b));
  }

  // XXX FAILS due to b not "dividing into" a properly
  // this matches cute's test faillure
  // this must not fulfill some precondition
  {
    strided_layout a(ns::int2(4,3));
    strided_layout b(ns::int2(4,3), ns::int2(3,1));

    assert(not test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(4,3), ns::int2(3,1));
    strided_layout b(ns::int2(4,3));

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(4,3), ns::int2(3,1));
    strided_layout b(12);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(4,3), ns::int2(3,1));
    strided_layout b(6, 2);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(4,5), ns::int2(3,0));
    strided_layout b(2, 2);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(2,4), ns::int2(2,5));
    strided_layout b(10, 1);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(4,3), ns::int2(3,1));
    strided_layout b(ns::int2(6,2), ns::int2(2,1));

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(2,4), ns::int2(2,5));
    strided_layout b(ns::int3(2,2,2), ns::int3(4,1,2));

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(8,8));

    auto b_shape = pair(ns::int3(2,2,2), ns::int3(2,2,2));
    auto b_stride = pair(ns::int3(1,16,4), ns::int3(8,2,32));

    strided_layout b(b_shape, b_stride);
  
    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(8,8), ns::int2(8,1));

    auto b_shape = pair(ns::int3(2,2,2), ns::int3(2,2,2));
    auto b_stride = pair(ns::int3(1,16,4), ns::int3(8,2,32));

    strided_layout b(b_shape, b_stride);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(4,2), ns::int2(1,16));
    strided_layout b(ns::int2(4,2), ns::int2(2, 1));

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(2,2), ns::int2(2,1));
    strided_layout b(ns::int2(2,2), ns::int2(2,1));

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int3(4,8,2));
    strided_layout b(ns::int3(2,2,2), ns::int3(2,8,1));

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int3(4,8,2), ns::int3(2,8,1));
    strided_layout b(ns::int3(2,2,2), ns::int3(1,8,2));

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int3(4,8,2), ns::int3(2,8,1));
    strided_layout b(ns::int3(4,2,2), ns::int3(2,8,1));

    assert(test_composition(a, b));
  }

  {
    strided_layout a(12,1);
    strided_layout b(4,1);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(12,3), ns::int2(1,24));
    strided_layout b(4,1);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(16, 2);
    strided_layout b(4, 2);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int3(128,24,5), ns::int3(1,128,3072));
    strided_layout b(64, 2);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int3(128,24,5), ns::int3(1,128,3072));
    strided_layout b(480, 32);

    assert(test_composition(a, b));
  }

  //// std::cout << "-------------------------------" << std::endl;
  //// std::cout << "cosize(b) > size(a) and divisibility" << std::endl;
  //// std::cout << "-------------------------------" << std::endl;

  {
    strided_layout a(1, 0);
    strided_layout b(4);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(1, 1);
    strided_layout b(4);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(4);
    strided_layout b(4, 2);

    assert(test_composition(a, b));
  }

  // Last mode gets extended
  {
    strided_layout a(ns::int2(4,3), ns::int2(3,1));
    strided_layout b(24);

    assert(test_composition(a, b));
  }

  // Last mode extension even without last mode divisibility
  {
    strided_layout a(ns::int2(4,3), ns::int2(3,1));
    strided_layout b(8);

    assert(test_composition(a, b));
  }

  // Capping a Layout with 1:0 forces divisibility and extends in stride-0
  {
    strided_layout a(ns::int3(4,3,1), ns::int3(3,1,0));
    strided_layout b(24);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(3, 1);
    strided_layout b(4, 1);

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int3(48,24,5), ns::int3(1,128,3072));
    strided_layout b(32, 1);

    assert(test_composition(a, b));
  }

  // XXX disable these until we decide to make ceil_div work with negative numbers

  //// std::cout << "-------------------------------" << std::endl;
  //// std::cout << "BETA: Negative strides"          << std::endl;
  //// std::cout << "-------------------------------" << std::endl;

  //{
  //  strided_layout a(4, -1);
  //  strided_layout b(4,  1);

  //  assert(test_composition(a, b));
  //}

  //{
  //  strided_layout a(4,  1);
  //  strided_layout b(4, -1);

  //  assert(test_composition(a, b));
  //}

  //{
  //  strided_layout a(4, -1);
  //  strided_layout b(4, -1);

  //  assert(test_composition(a, b));
  //}

  //{
  //  strided_layout a(4,  1);
  //  strided_layout b(4, -1);

  //  assert(test_composition(a, b));
  //}

  //{
  //  strided_layout a(ns::int2(4,4), ns::int2(-1,1));
  //  strided_layout b(ns::int3(2,4,2));

  //  assert(test_composition(a, b));
  //}

  //{
  //  strided_layout a(ns::int2(4,4),  ns::int2(-1,1));
  //  strided_layout b(ns::int3(2,4,2), ns::int3(1,4,2));

  //  assert(test_composition(a, b));
  //}

  //{
  //  strided_layout a(make_pair(1,ns::int2(2,4)), make_pair(0,ns::int2(-1,512)));
  //  strided_layout b(2, -1);

  //  assert(test_composition(a, b));
  //}

  //{
  //  strided_layout a(make_pair(1,ns::int2(2,4)), make_pair(0,ns::int2(-1,51)));
  //  strided_layout b(4, -1);

  //  assert(test_composition(a, b));
  //}

  {
    int num_rows = 16;
    int num_cols = 8;

    strided_layout a(ns::int2(num_rows*num_cols/2, 2), pair(ns::int2(0,1), ns::int2(1,0)));
    strided_layout b(ns::int2(num_rows,num_cols));
  
    assert(test_composition(a, b));
  }
  
  {

    strided_layout a(ns::int2(10,2), pair(ns::int2(0,1), ns::int2(1,0)));
    strided_layout b(ns::int2(5,4));

    assert(test_composition(a, b));
  }

  {
    strided_layout a(ns::int2(4,5), ns::int2(3,20));
    strided_layout b(2, 2);

    assert(test_composition(a,b));
  }
}

