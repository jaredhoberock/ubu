#include <algorithm>
#include <aspera/coordinate/point.hpp>

#undef NDEBUG
#include <cassert>

namespace ns = aspera;

template<class From, class To, class... Types>
void test(From arg1, Types... args)
{
  ns::point<From, 1 + sizeof...(args)> expected{arg1, args...};

  ns::point<To, 1 + sizeof...(args)> result{expected};

  assert(std::equal(expected.begin(), expected.end(), result.begin()));
}

void test_converting_constructor()
{
  test<char, unsigned char>(1);
  test<char, unsigned char>(1, 2);
  test<char, unsigned char>(1, 2, 3);
  test<char, unsigned char>(1, 2, 3, 4);

  test<unsigned char, char>(1);
  test<unsigned char, char>(1, 2);
  test<unsigned char, char>(1, 2, 3);
  test<unsigned char, char>(1, 2, 3, 4);

  test<short, unsigned short>(1);
  test<short, unsigned short>(1, 2);
  test<short, unsigned short>(1, 2, 3);
  test<short, unsigned short>(1, 2, 3, 4);

  test<unsigned short, short>(1);
  test<unsigned short, short>(1, 2);
  test<unsigned short, short>(1, 2, 3);
  test<unsigned short, short>(1, 2, 3, 4);

  test<int, unsigned int>(1);
  test<int, unsigned int>(1, 2);
  test<int, unsigned int>(1, 2, 3);
  test<int, unsigned int>(1, 2, 3, 4);

  test<unsigned int, int>(1);
  test<unsigned int, int>(1, 2);
  test<unsigned int, int>(1, 2, 3);
  test<unsigned int, int>(1, 2, 3, 4);

  test<std::size_t, int>(1);
  test<std::size_t, int>(1, 2);
  test<std::size_t, int>(1, 2, 3);
  test<std::size_t, int>(1, 2, 3, 4);

  test<int, std::size_t>(1);
  test<int, std::size_t>(1, 2);
  test<int, std::size_t>(1, 2, 3);
  test<int, std::size_t>(1, 2, 3, 4);

  test<float, int>(1);
  test<float, int>(1, 2);
  test<float, int>(1, 2, 3);
  test<float, int>(1, 2, 3, 4);

  test<int, float>(1);
  test<int, float>(1, 2);
  test<int, float>(1, 2, 3);
  test<int, float>(1, 2, 3, 4);

  test<float, double>(1);
  test<float, double>(1, 2);
  test<float, double>(1, 2, 3);
  test<float, double>(1, 2, 3, 4);

  test<double, float>(1);
  test<double, float>(1, 2);
  test<double, float>(1, 2, 3);
  test<double, float>(1, 2, 3, 4);

  test<int, double>(1);
  test<int, double>(1, 2);
  test<int, double>(1, 2, 3);
  test<int, double>(1, 2, 3, 4);

  test<double, int>(1);
  test<double, int>(1, 2);
  test<double, int>(1, 2, 3);
  test<double, int>(1, 2, 3, 4);
}

