#include <ubu/tensor/coordinates/point.hpp>

#undef NDEBUG
#include <cassert>

template<class T, class... Types>
void test(T arg1, Types... args)
{
  using namespace ubu;

  using point_t = point<T, 1 + sizeof...(args)>;

  point_t lhs(arg1, args...);
  point_t rhs(arg1, args...);

  point_t expected(arg1 - arg1, (args - args)...);

  assert(expected == lhs - rhs);
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

void test_binary_operator_minus()
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

