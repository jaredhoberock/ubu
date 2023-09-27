#include <algorithm>
#include <ubu/algorithm/for_each_n_after.hpp>
#include <ubu/causality/first_cause.hpp>
#include <ubu/execution/executor/associated_executor.hpp>
#include <ubu/memory/allocator/associated_allocator.hpp>
#include <ubu/memory/allocator/rebind_allocator.hpp>
#include <ubu/platform/cuda/device_allocator.hpp>
#include <ubu/platform/cuda/device_executor.hpp>
#include <ubu/platform/cuda/managed_allocator.hpp>

#undef NDEBUG
#include <cassert>

#include <random>
#include <vector>


namespace ns = ubu;


template<ns::executor E, ns::allocator A>
struct my_policy
{
  E executor;
  A allocator;

  my_policy(E e, A a)
    : executor(e), allocator(a)
  {}

  E associated_executor() const
  {
    return executor;
  }

  A associated_allocator() const
  {
    return allocator;
  }
};


template<class T, ns::execution_policy P>
void test_simple(P p)
{
  auto alloc = ns::rebind_allocator<T>(ns::associated_allocator(p));
  using alloc_type = decltype(alloc);
  using vector_type = std::vector<T, alloc_type>;

  vector_type input{5, alloc};
  vector_type output{7, 0, alloc};
  
  input[0] = 3; input[1] = 2; input[2] = 3; input[3] = 4; input[4] = 6;

  auto ready = ns::first_cause(ns::associated_executor(p));
  
  auto ptr = output.data();

  auto finished = ns::for_each_n_after(p, ready, input.begin(), input.size(), [ptr](int i)
  {
    ptr[i] = 1;
  });

  ns::wait(finished);
  
  assert(0 == output[0]);
  assert(0 == output[1]);
  assert(1 == output[2]);
  assert(1 == output[3]);
  assert(1 == output[4]);
  assert(0 == output[5]);
  assert(1 == output[6]);
}


template<class T, ns::execution_policy P>
void test(P p, std::size_t n)
{
  std::vector<T> h_input(n);

  std::default_random_engine rng;
  std::generate(h_input.begin(), h_input.end(), [&]() mutable
  {
    return rng() % n;
  });


  auto alloc = ns::rebind_allocator<T>(ns::associated_allocator(p));
  using alloc_type = decltype(alloc);
  using vector_type = std::vector<T, alloc_type>;

  vector_type d_input{h_input.begin(), h_input.end()};

  auto ready = ns::first_cause(ns::associated_executor(p));
  
  vector_type result(n, 0);
  auto ptr = result.data();
  auto finished = ns::for_each_n_after(p, ready, d_input.begin(), d_input.size(), [ptr](int i)
  {
    ptr[i] = 1;
  });

  ns::wait(finished);

  // compute the expected output
  std::vector<T> expected(n, 0);
  assert(expected.size() == n);
  auto h_ptr = expected.data();
  std::for_each_n(h_input.begin(), h_input.size(), [&](int i)
  {
    assert(i < expected.size());
    h_ptr[i] = 1;
  });

  assert(std::equal(expected.begin(), expected.end(), result.begin()));
}


void test_for_each_n_after()
{
//  {
//    // test_simple
//    
//    my_policy p{ns::cuda::device_executor{}, ns::cuda::managed_allocator<void>{}};
//
//    test_simple<std::int8_t>(p);
//    test_simple<std::uint8_t>(p);
//    test_simple<std::int16_t>(p);
//    test_simple<std::uint16_t>(p);
//    test_simple<std::int32_t>(p);
//    test_simple<std::uint32_t>(p);
//    test_simple<std::int64_t>(p);
//    test_simple<std::uint64_t>(p);
//
//  }
//
//  {
//    // test many sizes
//
//    my_policy p{ns::cuda::device_executor{}, ns::cuda::managed_allocator<void>{}};
//
//    for(std::size_t power = 0; power < 20; ++power)
//    {
//      test<int>(p, 1 << power);
//    }
//  }
}

