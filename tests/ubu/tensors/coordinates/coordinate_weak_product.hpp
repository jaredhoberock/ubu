#include <concepts>
#include <ubu/tensors/coordinates/coordinate_weak_product.hpp>
#include <ubu/utilities/constant.hpp>
#include <tuple>
#include <utility>

void test_coordinate_weak_product()
{
  using namespace ubu;

  {
    // int x int -> int
    
    constexpr auto a = 3;
    constexpr auto b = 2;
    constexpr auto expected = 6;
    constexpr auto result = coordinate_weak_product(a, b);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    // int x (int,int,int) -> (int,int,int)

    constexpr auto a = 2;
    constexpr auto b = std::tuple(1, 2, 3);
    constexpr auto expected = std::tuple(2, 4, 6);
    constexpr auto result = coordinate_weak_product(a, b);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    // (int,int,int) x (int,int,int) -> int

    constexpr auto a = std::tuple(4, 5, 6);
    constexpr auto b = std::tuple(1, 2, 3);
    constexpr auto expected = 32;
    constexpr auto result = coordinate_weak_product(a, b);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    // (int, (int,int) x (int, (int,int)) -> int

    constexpr auto a = std::tuple(4, std::pair(5, 6));
    constexpr auto b = std::tuple(1, std::pair(2, 3));
    constexpr auto expected = 32;
    constexpr auto result = coordinate_weak_product(a, b);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    // () x () -> 0

    constexpr auto a = std::tuple();
    constexpr auto b = std::tuple();
    constexpr auto expected = 0_c;
    constexpr auto result = coordinate_weak_product(a, b);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    // int x (int, (int,int)) -> (int, (int,int))

    constexpr auto a = 4;
    constexpr auto b = std::tuple(1, std::pair(2, 3));
    constexpr auto expected = std::tuple(4, std::pair(8, 12));
    constexpr auto result = coordinate_weak_product(a, b);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    // (int, (int, (int,int)) x (int, (int, (int,int)) -> int

    constexpr auto a        = std::tuple(5, std::pair(6, std::tuple(7, 8)));
    constexpr auto b        = std::tuple(1, std::pair(2, std::tuple(3, 4)));
    constexpr auto expected = 70;
    constexpr auto result   = coordinate_weak_product(a, b);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    //      ((1,2), (3,4)) x (((5,6), (7,8)), (9,10), (11,12)))

    constexpr auto a        = std::tuple(std::pair(1,2), std::pair(3,4));
    constexpr auto b        = std::tuple(std::pair(std::pair(5,6),std::pair(7,8)), std::pair(std::pair(9,10), std::pair(11,12)));
    constexpr auto expected = std::pair(90,100);
    constexpr auto result   = coordinate_weak_product(a,b);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    //     (1, (2,3)) x (1, ((2,2), (3,3)))
    // ->  (1*1 + (2,3)*((2,2), (3,3)))
    // ->  (1*1 + (2*(2,2) + 3*(3,3)))
    // ->  (1*1 + ((4,4) + (9,9)))
    // ->  (1*1 + (13,13))
    // ->  (1, (13,13))

    constexpr auto a         = std::tuple(1, std::pair(2,3));
    constexpr auto b         = std::tuple(1, std::pair(std::pair(2,2), std::pair(3,3)));
    constexpr auto expected  = std::tuple(1, std::pair(13,13));
    constexpr auto result    = coordinate_weak_product(a, b);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }


  {
    //    (1, (2,3), (4,5)) * (1, ((2,2), (3,3)), (4,5))
    //
    // -> (1*1 + (2,3)*((2,2),(3,3)) + (4,5)*(4,5))
    // -> (1*1 + (2*(2,2) + 3*(3,3)) + (4*4 + 5*5))
    // -> (1*1 + ((4,4) + (9,9)) + (16 + 25))
    // -> (1*1 + (13,13) + 41)
    // -> (1, (13,13), 41)

    constexpr auto a         = std::tuple(1, std::pair(2,3), std::pair(4,5));
    constexpr auto b         = std::tuple(1, std::tuple(std::pair(2,2), std::pair(3,3)), std::pair(4,5));
    constexpr auto expected  = std::tuple(1, std::pair(13,13), 41);
    constexpr auto result    = coordinate_weak_product(a, b);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    //    (5, (6,7)) * (1, (2, (3,4))) 
    //
    // -> (5*1, (6,7)*(2,(3,4)))
    // -> (5, (6*2, 7*(3,4)))
    // -> (5, (12, (21,28))

    constexpr auto a        = std::tuple(5, std::pair(6, 7));
    constexpr auto b        = std::tuple(1, std::pair(2, std::pair(3, 4)));
    constexpr auto expected = std::tuple(5, std::pair(12, std::pair(21,28)));

    constexpr auto result = coordinate_weak_product(a, b);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    //    ((), 2) x ((), 3))
    //
    // -> ()*() + 2*3
    // -> 0 + 6
    // -> 6

    constexpr auto unit = std::tuple();

    constexpr auto a = std::pair(unit, 2);
    constexpr auto b = std::pair(unit, 3);
    constexpr auto expected = 6;

    constexpr auto result = coordinate_weak_product(a,b);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    //    ((), (6, ((), 8))) x ((), (2, ((),4)))
    //
    // -> ()*() + (6,((),8))*(2,((),4))
    // -> 0 + (6*2 + ((),8)*((),4))
    // -> 0 + (12 + ()*() + 8*4)
    // -> 0 + (12 + 0 + 32)
    // -> 0 + 44
    // -> 44
    
    constexpr auto unit = std::tuple();

    constexpr auto a = std::tuple(unit, std::pair(6, std::tuple(8, unit)));
    constexpr auto b = std::tuple(unit, std::pair(2, std::tuple(4, unit)));
    constexpr auto expected = 44;

    constexpr auto result = coordinate_weak_product(a, b);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }

  {
    //    (1, (2,3), 4, 5) x (6, (7, (8,9)), 10, 11)
    //
    // -> 1*6 + (2*7,3*(8,9)) + 4*10 + 5*11
    // -> 6 + (14,(3*8,3*9)) + 40 + 55
    // -> 6 + (14,(24,27)) + 40 + 55
    // -> (6, (14,(24,27)) + 40 + 55
    // -> (6, (14,(24,27)), 40) + 55
    // -> (6, (14,(24,27)), 40, 55)

    constexpr auto a = std::tuple(1, std::pair(2,3), 4, 5);
    constexpr auto b = std::tuple(6, std::pair(7,std::pair(8,9)), 10, 11);
    constexpr auto expected = std::tuple(6, std::pair(14,std::pair(24,27)), 40, 55);

    constexpr auto result = coordinate_weak_product(a,b);

    static_assert(std::same_as<decltype(expected), decltype(result)>);
    static_assert(expected == result);
  }
}

