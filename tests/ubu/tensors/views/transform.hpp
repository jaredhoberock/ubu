#include <cassert>
#include <ranges>
#include <ubu/ubu.hpp>

void test_transform()
{
  using namespace ubu;

  {
    // test that we can transform a layout with a lambda

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

  {
    // test that we can transform a non-layout with a lambda

    std::vector<std::string> tensor = {"0", "1", "2"};

    auto integers = transform(tensor, [](const std::string& word)
    {
      return std::atoi(word.c_str());
    });

    std::vector<int> result;
    for(auto i : integers)
    {
      result.push_back(i);
    }

    std::vector<int> expected = {0, 1, 2};

    assert(std::ranges::equal(expected, result));
  }
}

