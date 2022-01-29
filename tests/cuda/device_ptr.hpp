#include <aspera/cuda/device_ptr.hpp>

#undef NDEBUG
#include <cassert>

namespace ns = aspera;


void test_device_ptr()
{
  using namespace ns;

  {
    // test default construction
    cuda::device_ptr<int> ptr;

    // silence "declared but never referenced" warnings
    static_cast<void>(ptr);
  }

  {
    // test construction from nullptr
    cuda::device_ptr<int> ptr{nullptr};

    assert(ptr.native_handle() == nullptr);
    assert(!ptr);
  }

  {
    // test writable device_ptr

    int* d_array{};
    assert(cudaMalloc(reinterpret_cast<void**>(&d_array), 4 * sizeof(int)) == cudaSuccess);
    int h_array[] = {0, 1, 2, 3};

    assert(cudaMemcpy(d_array, h_array, 4 * sizeof(int), cudaMemcpyDefault) == cudaSuccess);

    // test construction from raw pointer
    cuda::device_ptr<int> ptr{d_array};

    // test native_handle
    for(int i = 0; i < 4; ++i)
    {
      assert((ptr + i).native_handle() == d_array + i);
    }

    // test dereference
    for(int i = 0; i < 4; ++i)
    {
      assert(*(ptr + i) == h_array[i]);
    }

    // test subscript
    for(int i = 0; i < 4; ++i)
    {
      assert(ptr[i] == h_array[i]);
    }

    // test store
    for(int i = 0; i < 4; ++i)
    {
      ptr[i] = 4 - i;
    }

    assert(cudaMemcpy(h_array, d_array, 4 * sizeof(int), cudaMemcpyDefault) == cudaSuccess);
    for(int i = 0; i < 4; ++i)
    {
      assert(h_array[i] == 4 - i);
    }

    assert(cudaFree(d_array) == cudaSuccess);
  }

  {
    // test readable device_ptr

    int* d_array{};
    assert(cudaMalloc(reinterpret_cast<void**>(&d_array), 4 * sizeof(int)) == cudaSuccess);
    int h_array[] = {0, 1, 2, 3};

    assert(cudaMemcpy(d_array, h_array, 4 * sizeof(int), cudaMemcpyDefault) == cudaSuccess);

    // test construction from raw pointer
    cuda::device_ptr<const int> ptr{d_array};

    // test native_handle
    for(int i = 0; i < 4; ++i)
    {
      assert((ptr + i).native_handle() == d_array + i);
    }

    // test dereference
    for(int i = 0; i < 4; ++i)
    {
      assert(*(ptr + i) == h_array[i]);
    }

    // test subscript
    for(int i = 0; i < 4; ++i)
    {
      assert(ptr[i] == h_array[i]);
    }

    assert(cudaFree(d_array) == cudaSuccess);
  }
}

