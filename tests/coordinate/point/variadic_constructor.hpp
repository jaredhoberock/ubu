#include <algorithm>
#include <ubu/coordinate/point.hpp>

#undef NDEBUG
#include <cassert>

namespace ns = ubu;


template<class T, class... Types>
void test(T arg1, Types... args)
{
  ns::point<T, 1 + sizeof...(args)> p(arg1, args...);

  std::array<T, 1 + sizeof...(args)> expected{arg1, static_cast<T>(args)...};

  assert(std::equal(expected.begin(), expected.end(), p.begin()));
}


void test_variadic_constructor()
{
  test<char>(1);
  test<char>(1, 2);
  test<char>(1, 2, 3);
  test<char>(1, 2, 3, 4);

  test<unsigned char>(1);
  test<unsigned char>(1, 2);
  test<unsigned char>(1, 2, 3);
  test<unsigned char>(1, 2, 3, 4);

  test<short>(1);
  test<short>(1, 2);
  test<short>(1, 2, 3);
  test<short>(1, 2, 3, 4);

  test<unsigned short>(1);
  test<unsigned short>(1, 2);
  test<unsigned short>(1, 2, 3);
  test<unsigned short>(1, 2, 3, 4);

  test<int>(1);
  test<int>(1, 2);
  test<int>(1, 2, 3);
  test<int>(1, 2, 3, 4);

  test<unsigned int>(1);
  test<unsigned int>(1, 2);
  test<unsigned int>(1, 2, 3);
  test<unsigned int>(1, 2, 3, 4);

  test<std::size_t>(1);
  test<std::size_t>(1, 2);
  test<std::size_t>(1, 2, 3);
  test<std::size_t>(1, 2, 3, 4);

  test<float>(1);
  test<float>(1, 2);
  test<float>(1, 2, 3);
  test<float>(1, 2, 3, 4);

  test<double>(1);
  test<double>(1, 2);
  test<double>(1, 2, 3);
  test<double>(1, 2, 3, 4);
}

