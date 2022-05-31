#include <ubu/property/require_property.hpp>
#include <ubu/property/property.hpp>
#include <cassert>

namespace ns = ubu;


template<class>
concept any_type = true;

#if defined(__circle_lang__)
constexpr ns::property<"test", int, any_type> test;
#else
DEFINE_PROPERTY_TEMPLATE(test, any_type);
test_property<int> test;
#endif


struct has_test
{
  int test_;

  int test() const
  {
    return test_;
  }

  void test(int value)
  {
    test_ = value;
  }
};

static_assert(ns::has_property<has_test, decltype(test)>);


struct can_mix_test
{
  has_test mix_test(int value) const
  {
    return has_test{value};
  }

  has_test mix(decltype(test) prop) const
  {
    return has_test{prop.value};
  }
};

static_assert(ns::can_mix_property<can_mix_test, decltype(test)>);


void test_require_property()
{
  has_test already_has;
  already_has = ns::require_property(already_has, test(13));
  assert(test(already_has) == 13);

  can_mix_test can_mix;
  has_test mixed = ns::require_property(can_mix, test(7));
  assert(test(mixed) == 7);
}

