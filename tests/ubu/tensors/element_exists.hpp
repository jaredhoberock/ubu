#include <array>
#include <cassert>
#include <ubu/tensors/element_exists.hpp>
#include <ubu/tensors/coordinates/point.hpp>
#include <utility>
#include <vector>


namespace ns = ubu;


void test_element_exists()
{
  // test 2-arrays
  {
    std::array<int, 2> x{13, 7};
    assert(true == ns::element_exists(x, 0));
    assert(true == ns::element_exists(x, 1));
  }
  
  // test point
  {
    ns::point<int,3> x{13, 7, 42};
    assert(true == ns::element_exists(x, 0));
    assert(true == ns::element_exists(x, 1));
    assert(true == ns::element_exists(x, 2));
  }

  // test typedefs
  {
    ns::int2 x{13, 7};
    assert(true == ns::element_exists(x, 0));
    assert(true == ns::element_exists(x, 1));
  }

  // test nested std::vector with 1d coord
  {
    std::vector<std::vector<int>> nested_vec({{13,7}, {42,66}});
    assert(true == ns::element_exists(nested_vec, 0));
    assert(true == ns::element_exists(nested_vec, 1));
  }

  // test nested std::vector with 2d coord
  {
    std::vector<std::vector<int>> nested_vec({{13,7}, {42,66}});

    assert(true == ns::element_exists(nested_vec, std::pair(0,0)));
    assert(true == ns::element_exists(nested_vec, std::pair(1,0)));
    assert(true == ns::element_exists(nested_vec, std::pair(0,1)));
    assert(true == ns::element_exists(nested_vec, std::pair(1,1)));

    assert(false == ns::element_exists(nested_vec, std::pair(2,0)));
    assert(false == ns::element_exists(nested_vec, std::pair(2,1)));

    // XXX these should also return false
    //     we need to fold the in_domain check into terminal_element_exists
    //     to make this work
    // assert(false == ns::element_exists(nested_vec, std::pair(0,2)));
    // assert(false == ns::element_exists(nested_vec, std::pair(1,2)));
  }

  // test lambda
  {
    auto lambda = [](int){ return 13; };

    assert(true == ns::element_exists(lambda, 0));
  }
}

