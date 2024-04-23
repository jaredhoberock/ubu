#include <cassert>
#include <concepts>
#include <iostream>
#include <ubu/tensor/fancy_span.hpp>
#include <ubu/miscellaneous/bounded.hpp>

void test_fancy_span()
{
  using namespace ubu;

  std::vector<int> vec(100);

  {
    fancy_span<int*> s;
    
    static_assert(std::same_as<const std::size_t, decltype(s.extent)>);
    static_assert(s.extent == std::dynamic_extent);
    static_assert(s.extent == std::numeric_limits<std::size_t>::max());
  }

  {
    fancy_span<int*> s(vec);

    static_assert(std::same_as<const std::size_t, decltype(s.extent)>);
    static_assert(s.extent == std::dynamic_extent);
    static_assert(s.extent == std::numeric_limits<std::size_t>::max());
  }

  {
    fancy_span<int*,constant<10>> s(vec.data(), 10_c);

    static_assert(std::same_as<const constant<10>, decltype(s.extent)>);
    static_assert(s.extent == 10);
    static_assert(s.extent == 10_c);
  }

  {
    fancy_span<int*,int> s(vec.data(), 10);

    static_assert(std::same_as<const int, decltype(s.extent)>);
    static_assert(s.extent == std::numeric_limits<int>::max());
  }

  {
    fancy_span s(vec.data(), 10_c);

    static_assert(std::same_as<const constant<10>, decltype(s.extent)>);
    static_assert(s.extent == 10);
    static_assert(s.extent == 10_c);
    static_assert(std::same_as<const constant<10>, decltype(s.extent)>);

    auto size = s.size();
    static_assert(std::same_as<constant<10>, decltype(size)>);
  }

  {
    fancy_span<int*,bounded<10>> s(vec.data(), 5);

    static_assert(std::same_as<const constant<10>, decltype(s.extent)>);
    static_assert(s.extent == 10);
    static_assert(s.extent == bounded<10>(10));
    static_assert(s.extent == bounded<10>::bound);
    static_assert(s.extent == std::numeric_limits<bounded<10>>::max());

    auto size = s.size();
    assert(size == bounded<10>(5));
    static_assert(std::same_as<bounded<10>, decltype(size)>);
  }

  {
    bounded sz(5, 10_c);
    fancy_span s(vec.data(), sz);
    static_assert(std::same_as<const constant<10>, decltype(s.extent)>);

    auto size = s.size();
    static_assert(std::same_as<bounded<10>, decltype(size)>);
  }

  {
    fancy_span s(vec.data(), 10_c);
    static_assert(std::same_as<const constant<10>, decltype(s.extent)>);

    auto size = s.size();
    static_assert(std::same_as<constant<10>, decltype(size)>);
  }

  {
    fancy_span s(vec);

    auto first_10 = s.first(10);
    using expected_type = fancy_span<int*,std::size_t>;

    static_assert(std::same_as<expected_type, decltype(first_10)>);
  }

  {
    fancy_span s(vec.data(), 100_c);

    auto first_10 = s.first(10);
    using expected_type = fancy_span<int*,int>;

    static_assert(std::same_as<expected_type, decltype(first_10)>);
  }

  {
    fancy_span s(vec.data(), 100_c);

    auto first_10 = s.first(10_c);
    using expected_type = fancy_span<int*,constant<10>>;

    static_assert(std::same_as<expected_type, decltype(first_10)>);
  }

  {
    fancy_span s(vec.data(), bounded(50, 100_c));

    auto first_10 = s.first(10);
    using expected_type = fancy_span<int*,int>;

    static_assert(std::same_as<expected_type, decltype(first_10)>);
  }

  {
    fancy_span s(vec.data(), bounded(50, 100_c));

    auto first_10 = s.first(bounded(10, 20_c));
    using expected_type = fancy_span<int*,bounded<20>>;

    static_assert(std::same_as<expected_type, decltype(first_10)>);
  }

  {
    fancy_span s(vec);

    auto last_10 = s.last(10);
    using expected_type = fancy_span<int*,std::size_t>;

    static_assert(std::same_as<expected_type, decltype(last_10)>);
  }

  {
    fancy_span s(vec.data(), 100_c);

    auto last_10 = s.last(10);
    using expected_type = fancy_span<int*,int>;

    static_assert(std::same_as<expected_type, decltype(last_10)>);
  }

  {
    fancy_span s(vec.data(), 100_c);

    auto last_10 = s.last(10_c);
    using expected_type = fancy_span<int*,constant<10>>;

    static_assert(std::same_as<expected_type, decltype(last_10)>);
  }

  {
    fancy_span s(vec.data(), bounded(50, 100_c));

    auto last_10 = s.last(10);
    using expected_type = fancy_span<int*,int>;

    static_assert(std::same_as<expected_type, decltype(last_10)>);
  }

  {
    fancy_span s(vec.data(), bounded(50, 100_c));

    auto last_10 = s.last(bounded(10, 20_c));
    using expected_type = fancy_span<int*,bounded<20>>;

    static_assert(std::same_as<expected_type, decltype(last_10)>);
  }
}

