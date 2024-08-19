#include <concepts>
#include <ubu/tensors/coordinates/colexicographical_lift.hpp>
#include <ubu/tensors/coordinates/point.hpp>
#include <tuple>
#include <utility>

void test_colexicographical_lift()
{
  using namespace ubu;

  {
    // () -> ()

    constexpr auto coord = std::tuple();
    constexpr auto shape = std::tuple();
    constexpr auto expected = std::tuple();
    constexpr auto result = colexicographical_lift(coord, shape);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    // ((), ()) -> ((), ())

    constexpr auto coord = std::tuple(std::tuple(), std::tuple());
    constexpr auto shape = std::tuple(std::tuple(), std::tuple());
    constexpr auto expected = std::tuple(std::tuple(), std::tuple());
    constexpr auto result = colexicographical_lift(coord, shape);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    // int -> int

    constexpr auto coord = 5;
    constexpr auto shape = 10;
    constexpr auto expected = 5;
    constexpr auto result = colexicographical_lift(coord, shape);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    // int -> (int)

    constexpr auto coord = 5;
    constexpr auto shape = std::tuple(10);
    constexpr auto expected = 5; // XXX this is a little unexpected
    constexpr auto result = colexicographical_lift(coord, shape);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    // (int) -> int

    constexpr auto coord = std::tuple(5);
    constexpr auto shape = 10;
    constexpr auto expected = 5;

    constexpr auto result = colexicographical_lift(coord, shape);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    // int -> (int, int)

    constexpr auto coord = 23;
    constexpr auto shape = std::pair(5,5);
    constexpr auto expected = std::pair(3,4);
    constexpr auto result = colexicographical_lift(coord, shape);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    // int -> (int, int, int)

    constexpr auto coord = 1234;
    constexpr auto shape = ubu::int3(2,3,5);
    constexpr auto expected = ubu::int3(0,2,205);
    constexpr auto result = colexicographical_lift(coord, shape);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    // int -> (int, int, int, int)

    constexpr auto coord = 1234;
    constexpr auto shape = ubu::uint4(2,3,5,7);
    constexpr auto expected = ubu::uint4(0,2,0,41);
    constexpr auto result = colexicographical_lift(coord, shape);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    // (int, (int, (int,int))) -> ((int,int), ((int,int,int), (int,(int,int,int,int))))

    constexpr auto coord    = std::tuple(1, std::tuple(2, std::tuple(3,4)));
    constexpr auto shape    = std::tuple(std::pair(1,2), std::tuple(ubu::int3(2,2,2), std::tuple(3,ubu::int4(4,4,4,4))));
    constexpr auto expected = std::tuple(std::pair(0,1), std::tuple(ubu::int3(0,1,0), std::tuple(3,ubu::int4(0,1,0,0))));
    constexpr auto result   = colexicographical_lift(coord, shape);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  return;
}

