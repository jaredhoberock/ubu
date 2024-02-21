#include <ubu/tensor/coordinate/concepts/superdimensional.hpp>
#include <array>
#include <tuple>
#include <utility>

void test_superdimensional()
{
  using namespace ubu;

  // test some scalars
  static_assert(superdimensional<int,int>);
  static_assert(superdimensional<int,unsigned int>);
  static_assert(superdimensional<unsigned int,int>);
  static_assert(superdimensional<char, int>);
  static_assert(superdimensional<std::array<int,1>, int>);

  {
    // test some coordinates that are weakly_congruent (in the other direction)

    {
      using T1 = std::pair<int,int>;
      using T2 = int;
      static_assert(superdimensional<T1,T2>);
    }

    {
      using T1 = std::tuple<int,int,int>;
      using T2 = int;
      static_assert(superdimensional<T1,T2>);
    }

    {
      using T1 = std::pair<int,std::pair<int,int>>;
      using T2 = std::pair<int,int>;
      static_assert(superdimensional<T1,T2>);
    }

    {
      using T1 = std::pair<std::tuple<int,int,int>, std::tuple<int,int,int>>;
      using T2 = std::pair<int,int>;
      static_assert(superdimensional<T1,T2>);
    }

    {
      // mix in some references
      using T1 = std::pair<std::tuple<int,int,int>&, std::tuple<int,int,int>>;
      using T2 = std::pair<int,int>&;
      static_assert(superdimensional<T1,T2>);
    }
  }

  {
    {
      using T1 = std::tuple<std::tuple<int,int,int>, std::tuple<int,int,int>, int>;
      using T2 = std::pair<int,int>;
      static_assert(superdimensional<T1,T2>);
    }

    {
      using T1 = std::tuple<int,int,int,int>;
      using T2 = std::tuple<int,int,int>;
      static_assert(superdimensional<T1,T2>);
    }

    {
      // mix in some references
      using T1 = volatile std::tuple<int,int&&,int,int>&;
      using T2 = std::tuple<int&,int,const int&>;
      static_assert(superdimensional<T1,T2>);
    }
  }
}

