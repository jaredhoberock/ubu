#include <algorithm>
#include <ubu/memory/pointer/remote_ptr.hpp>

#undef NDEBUG
#include <cassert>
#include <cstring>

namespace ns = ubu;


struct trivial_loader
{
  using address_type = void*;

  void upload(const void* from, std::size_t num_bytes, address_type to) const
  {
    std::memcpy(to, from, num_bytes);
  }

  void download(address_type from, std::size_t num_bytes, void* to) const
  {
    std::memcpy(to, from, num_bytes);
  }

  constexpr bool operator==(const trivial_loader&) const
  {
    return true;
  }
};


void test_remote_ptr()
{
  using namespace ns;

  {
    // test concepts
    static_assert(std::random_access_iterator<remote_ptr<int, trivial_loader>>);
  }

  {
    // test default construction
    remote_ptr<int, trivial_loader> ptr;

    // silence "declared but never referenced" warnings
    static_cast<void>(ptr);
  }

  {
    // test construction from nullptr
    remote_ptr<int, trivial_loader> ptr{nullptr};

    assert(ptr.to_address() == nullptr);
    assert(!ptr);
  }

  {
    int array[] = {0, 1, 2, 3};

    // test construction from raw pointer
    remote_ptr<int, trivial_loader> ptr(array);

    // test native_handle
    for(int i = 0; i < 4; ++i)
    {
      assert((ptr + i).to_address() == &array[i]);
    }

    // test dereference
    for(int i = 0; i < 4; ++i)
    {
      assert(*(ptr + i) == array[i]);
    }

    // test subscript
    for(int i = 0; i < 4; ++i)
    {
      assert(ptr[i] == array[i]);
    }

    // test store
    for(int i = 0; i < 4; ++i)
    {
      ptr[i] = 4 - i;
    }

    for(int i = 0; i < 4; ++i)
    {
      assert(array[i] == 4 - i);
    }
  }

  {
    int array[] = {0, 1, 2, 3};

    // test construction from raw pointer
    remote_ptr<const int, trivial_loader> ptr(array);

    // test native_handle
    for(int i = 0; i < 4; ++i)
    {
      assert((ptr + i).to_address() == &array[i]);
    }

    // test dereference
    for(int i = 0; i < 4; ++i)
    {
      assert(*(ptr + i) == array[i]);
    }

    // test subscript
    for(int i = 0; i < 4; ++i)
    {
      assert(ptr[i] == array[i]);
    }
  }
}

