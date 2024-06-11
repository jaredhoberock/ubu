#include <cassert>
#include <functional>
#include <iostream>
#include <numeric>
#include <ranges>
#include <ubu/tensor/coordinates/point.hpp>
#include <ubu/tensor/iterators.hpp>
#include <ubu/tensor/views/enumerate.hpp>
#include <ubu/tensor/views/lattice.hpp>
#include <vector>

void test_enumerate()
{
  using namespace ubu;

  {
    std::vector<int> vec(13);
    std::iota(vec.begin(), vec.end(), 0);

    for(auto [coord, value] : enumerate(vec))
    {
      assert(coord == value);
      value = -coord;
    }

    using namespace std::views;

    auto expected = iota(0,13) | transform(std::negate());

    assert(std::ranges::equal(expected, vec));
  }

  {
    lattice<ubu::int3> lat;

    for(auto [coord, value] : enumerate(lat))
    {
      assert(coord == value);
    }
  }
}

