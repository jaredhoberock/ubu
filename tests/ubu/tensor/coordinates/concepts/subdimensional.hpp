#include <ubu/tensor/coordinates/concepts/subdimensional.hpp>
#include <ubu/tensor/coordinates/concepts/weakly_congruent.hpp>
#include <array>
#include <tuple>
#include <utility>

void test_subdimensional()
{
  using namespace ubu;

  // test some scalars
  static_assert(subdimensional<int,int>);
  static_assert(subdimensional<int,unsigned int>);
  static_assert(subdimensional<unsigned int,int>);
  static_assert(subdimensional<char, int>);
  static_assert(subdimensional<std::array<int,1>, int>);

  {
    // test some coordinates that are weakly_congruent

    {
      using T1 = int;
      using T2 = std::pair<int,int>;
      static_assert(subdimensional<T1,T2>);
      static_assert(weakly_congruent<T1,T2>);
    }

    {
      using T1 = int;
      using T2 = std::tuple<int,int,int>;
      static_assert(subdimensional<T1,T2>);
      static_assert(weakly_congruent<T1,T2>);
    }

    {
      using T1 = std::pair<int,int>;
      using T2 = std::pair<int,std::pair<int,int>>;
      static_assert(subdimensional<T1,T2>);
      static_assert(weakly_congruent<T1,T2>);
    }

    {
      using T1 = std::pair<int,int>;
      using T2 = std::pair<std::tuple<int,int,int>, std::tuple<int,int,int>>;
      static_assert(subdimensional<T1,T2>);
      static_assert(weakly_congruent<T1,T2>);
    }

    {
      // mix in some references
      using T1 = std::pair<int,int>&;
      using T2 = std::pair<std::tuple<int,int,int>&, std::tuple<int,int,int>>;
      static_assert(subdimensional<T1,T2>);
      static_assert(weakly_congruent<T1,T2>);
    }
  }

  {
    // test some subdimensional coordinates that are not weakly_congruent

    {
      using T1 = std::pair<int,int>;
      using T2 = std::tuple<std::tuple<int,int,int>, std::tuple<int,int,int>, int>;
      static_assert(subdimensional<T1,T2>);
      static_assert(not weakly_congruent<T1,T2>);
    }

    {
      using T1 = std::tuple<int,int,int>;
      using T2 = std::tuple<int,int,int,int>;
      static_assert(subdimensional<T1,T2>);
      static_assert(not weakly_congruent<T1,T2>);
    }

    {
      // mix in some references
      using T1 = std::tuple<int&,int,const int&>;
      using T2 = volatile std::tuple<int,int&&,int,int>&;
      static_assert(subdimensional<T1,T2>);
      static_assert(not weakly_congruent<T1,T2>);
    }
  }
}

