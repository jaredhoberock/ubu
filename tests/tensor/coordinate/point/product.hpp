#include <ubu/tensor/coordinate/point.hpp>

#undef NDEBUG
#include <cassert>

namespace ns = ubu;

template<class T, class... Args>
void test(Args... args)
{
  using point_t = ns::point<T, sizeof...(args)>;

  point_t x(args...);

  T expected = (... * args);

  assert(expected == x.product());
}

template<class T>
void test()
{
  test<T>(1);
  test<T>(1, 2);
  test<T>(1, 2, 3);
  test<T>(1, 2, 3, 4);
  test<T>(1, 2, 3, 4, 5);
  test<T>(1, 2, 3, 4, 5, 6);
  test<T>(1, 2, 3, 4, 5, 6, 7);
  test<T>(1, 2, 3, 4, 5, 6, 7, 8);
  test<T>(1, 2, 3, 4, 5, 6, 7, 8, 9);
  test<T>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
}

void test_product()
{
  test<char>();
  test<unsigned char>();
  test<short>();
  test<unsigned short>();
  test<int>();
  test<unsigned int>();
  test<std::size_t>();
  test<float>();
  test<double>();
}

