#include <cassert>
#include <ranges>
#include <ubu/ubu.hpp>

void test_transform()
{
  using namespace ubu;

  {
    // test that we

    ubu::int2 shape(2,3);
    lattice tensor(shape);

    auto summed = transform(tensor, [](auto e)
    {
      return e.x + e.y;
    });

    std::vector<int> result;
    for(auto sum : summed)
    {
      result.push_back(sum);
    }

    std::vector<int> expected;
    for(auto e : tensor)
    {
      expected.push_back(e.x + e.y);
    }

    assert(std::ranges::equal(expected, result));
  }
}

