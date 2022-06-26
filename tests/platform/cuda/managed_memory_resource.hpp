#include <ubu/platform/cuda/managed_memory_resource.hpp>

#undef NDEBUG
#include <cassert>


namespace ns = ubu;


void test_allocate()
{
  using namespace ns;

  cuda::managed_memory_resource r;

  int* ptr = static_cast<int*>(r.allocate(sizeof(int)));

  cudaPointerAttributes attr{};
  assert(cudaSuccess == cudaPointerGetAttributes(&attr, ptr));
  assert(attr.type == cudaMemoryTypeManaged);
  assert(attr.device == r.device());
  assert(attr.devicePointer == ptr);
  assert(attr.hostPointer == ptr);

  int expected = 13;
  *ptr = expected;
  int result = *ptr;

  assert(expected == result);

  r.deallocate(ptr, sizeof(int));
}


void test_comparison()
{
  using namespace ns;

  cuda::managed_memory_resource r0{0};

  // compare a resource against itself
  assert(r0.is_equal(r0));
  assert(r0 == r0);
  assert(!(r0 != r0));

  // compare resources of the same device
  cuda::managed_memory_resource other_r0{0};
  assert(r0.is_equal(other_r0));
  assert(r0 == other_r0);
  assert(!(r0 != other_r0));

  // resources of different devices compare different
  cuda::managed_memory_resource r1{1};
  assert(!r0.is_equal(r1));
  assert(!(r0 == r1));
  assert(r0 != r1);
}


void test_copy_construction()
{
  using namespace ns;

  cuda::managed_memory_resource r{0};
  cuda::managed_memory_resource r_copy = r;

  assert(r == r_copy);
}


void test_device()
{
  using namespace ns;

  cuda::managed_memory_resource r0;
  assert(0 == r0.device());

  cuda::managed_memory_resource r1{1};
  assert(1 == r1.device());
}


void test_throw_on_failure()
{
  using namespace ns;

  cuda::managed_memory_resource r;

  try
  {
    std::size_t num_bytes = -1;
    r.allocate(num_bytes);
  }
  catch(...)
  {
    return;
  }

  assert(0);
}


void test_managed_memory_resource()
{
  test_allocate();
  test_comparison();
  test_copy_construction();
  test_device();
  test_throw_on_failure();
}

