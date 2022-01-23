#include <aspera/coordinate/point.hpp>
#include <cassert>
#include <sstream>
#include <string>

template<class Arg>
void output_all(std::ostream& os, const std::string&, Arg arg)
{
  os << arg;
}

template<class Arg, class... Args>
void output_all(std::ostream& os, const std::string& delimiter, Arg arg1, Args... args)
{
  os << arg1;
  os << delimiter;

  output_all(os, delimiter, args...);
}

template<class T, class... Types>
void test(Types... args)
{
  std::ostringstream expected;
  expected << "{";
  output_all(expected, ", ", static_cast<T>(args)...);
  expected << "}";

  using namespace aspera;

  point<T, sizeof...(args)> x(static_cast<T>(args)...);

  std::ostringstream output;
  output << x;

  assert(expected.str() == output.str());
}

void test_output_operator()
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

