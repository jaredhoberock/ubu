#include <aspera/coordinate/coordinate.hpp>
#include <aspera/coordinate/point.hpp>

void test_coordinate()
{
  using namespace aspera;

  static_assert(coordinate<point<char,1>>);
  static_assert(coordinate<point<char,2>>);
  static_assert(coordinate<point<char,3>>);
  static_assert(coordinate<point<char,4>>);

  static_assert(coordinate<point<unsigned char,1>>);
  static_assert(coordinate<point<unsigned char,2>>);
  static_assert(coordinate<point<unsigned char,3>>);
  static_assert(coordinate<point<unsigned char,4>>);

  static_assert(coordinate<point<short,1>>);
  static_assert(coordinate<point<short,2>>);
  static_assert(coordinate<point<short,3>>);
  static_assert(coordinate<point<short,4>>);

  static_assert(coordinate<point<unsigned short,1>>);
  static_assert(coordinate<point<unsigned short,2>>);
  static_assert(coordinate<point<unsigned short,3>>);
  static_assert(coordinate<point<unsigned short,4>>);

  static_assert(coordinate<point<int,1>>);
  static_assert(coordinate<point<int,2>>);
  static_assert(coordinate<point<int,3>>);
  static_assert(coordinate<point<int,4>>);

  static_assert(coordinate<point<unsigned int,1>>);
  static_assert(coordinate<point<unsigned int,2>>);
  static_assert(coordinate<point<unsigned int,3>>);
  static_assert(coordinate<point<unsigned int,4>>);

  static_assert(coordinate<point<std::size_t,1>>);
  static_assert(coordinate<point<std::size_t,2>>);
  static_assert(coordinate<point<std::size_t,3>>);
  static_assert(coordinate<point<std::size_t,4>>);

  static_assert(coordinate<point<float,1>>);
  static_assert(coordinate<point<float,2>>);
  static_assert(coordinate<point<float,3>>);
  static_assert(coordinate<point<float,4>>);

  static_assert(coordinate<point<double,1>>);
  static_assert(coordinate<point<double,2>>);
  static_assert(coordinate<point<double,3>>);
  static_assert(coordinate<point<double,4>>);
}

