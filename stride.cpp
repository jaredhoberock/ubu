#include "stride.hpp"

#undef NDEBUG
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

int main()
{
  std::vector<int> v(10);
  std::iota(v.begin(), v.end(), 0);

  auto all = std::views::all(v);
  auto strided = stride(all, 2);

  std::cout << "size: " << std::ranges::size(strided) << std::endl;

  int sum = 0;
  for(auto x : stride(v, 2))
  {
    sum += x;
  }

  assert(20 == sum);

  std::cout << "OK" << std::endl;

  return 0;
}

