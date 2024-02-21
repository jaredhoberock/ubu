#include <ubu/tensor/coordinate/concepts/congruent.hpp>
#include <ubu/tensor/coordinate/concepts/strictly_subdimensional.hpp>
#include <ubu/tensor/coordinate/concepts/weakly_congruent.hpp>
#include <array>
#include <tuple>
#include <utility>

void test_strictly_subdimensional()
{
  using namespace ubu;

  // test some scalars
  static_assert(not strictly_subdimensional<int,int>);
  static_assert(not strictly_subdimensional<int,unsigned int>);
  static_assert(not strictly_subdimensional<unsigned int,int>);
  static_assert(not strictly_subdimensional<char, int>);
  static_assert(not strictly_subdimensional<std::array<int,1>, int>);

  {
    // test some coordinates that are congruent

    {
      using T1 = int;
      using T2 = unsigned int;
      static_assert(not strictly_subdimensional<T1,T2>);
      static_assert(congruent<T1,T2>);
    }

    {
      using T1 = std::tuple<int, std::pair<unsigned int, char>>;
      using T2 = std::pair<unsigned int, std::array<char,2>>;
      static_assert(not strictly_subdimensional<T1,T2>);
      static_assert(congruent<T1,T2>);
    }

    {
      // mix in some references
      using T1 = std::tuple<int, std::pair<unsigned int&&, char&&>>;
      using T2 = std::pair<const unsigned int&, std::array<char,2>>;
      static_assert(not strictly_subdimensional<T1,T2>);
      static_assert(congruent<T1,T2>);
    }
  }

  {
    // test some coordinates that are weakly_congruent

    {
      using T1 = int;
      using T2 = std::pair<int,int>;
      static_assert(strictly_subdimensional<T1,T2>);
      static_assert(weakly_congruent<T1,T2>);
    }

    {
      using T1 = int;
      using T2 = std::tuple<int,int,int>;
      static_assert(strictly_subdimensional<T1,T2>);
      static_assert(weakly_congruent<T1,T2>);
    }

    {
      using T1 = std::pair<int,int>;
      using T2 = std::pair<int,std::pair<int,int>>;
      static_assert(strictly_subdimensional<T1,T2>);
      static_assert(weakly_congruent<T1,T2>);
    }

    {
      using T1 = std::pair<int,int>;
      using T2 = std::pair<std::tuple<int,int,int>, std::tuple<int,int,int>>;
      static_assert(strictly_subdimensional<T1,T2>);
      static_assert(weakly_congruent<T1,T2>);
    }

    {
      // mix in some references
      using T1 = std::pair<int,int>&;
      using T2 = std::pair<std::tuple<int,int,int>&, std::tuple<int,int,int>>;
      static_assert(strictly_subdimensional<T1,T2>);
      static_assert(weakly_congruent<T1,T2>);
    }
  }

  {
    // test some subdimensional coordinates that are not weakly_congruent

    {
      using T1 = std::pair<int,int>;
      using T2 = std::tuple<std::tuple<int,int,int>, std::tuple<int,int,int>, int>;
      static_assert(strictly_subdimensional<T1,T2>);
      static_assert(not weakly_congruent<T1,T2>);
    }

    {
      using T1 = std::tuple<int,int,int>;
      using T2 = std::tuple<int,int,int,int>;
      static_assert(strictly_subdimensional<T1,T2>);
      static_assert(not weakly_congruent<T1,T2>);
    }

    {
      // mix in some references
      using T1 = std::tuple<int&,int,const int&>;
      using T2 = volatile std::tuple<int,int&&,int,int>&;
      static_assert(strictly_subdimensional<T1,T2>);
      static_assert(not weakly_congruent<T1,T2>);
    }
  }
}

