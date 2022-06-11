#include <algorithm>
#include <ubu/memory/fancy_ptr.hpp>

#undef NDEBUG
#include <cassert>

namespace ns = ubu;


struct trivial_copier
{
  template<class T>
  using address = T*;

  template<class T>
  void copy_n(const T* from, std::size_t count, T* to) const
  {
    std::copy_n(from, count, to);
  }

  constexpr bool operator==(const trivial_copier&) const
  {
    return true;
  }
};


void test_fancy_ptr()
{
  using namespace ns;

  {
    // test concepts
    static_assert(std::random_access_iterator<fancy_ptr<int, trivial_copier>>);
  }

  {
    // test default construction
    fancy_ptr<int, trivial_copier> ptr;

    // silence "declared but never referenced" warnings
    static_cast<void>(ptr);
  }

  {
    // test construction from nullptr
    fancy_ptr<int, trivial_copier> ptr{nullptr};

    assert(ptr.to_address() == nullptr);
    assert(!ptr);
  }

  {
    int array[] = {0, 1, 2, 3};

    // test construction from raw pointer
    fancy_ptr<int, trivial_copier> ptr(array);

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
    fancy_ptr<const int, trivial_copier> ptr(array);

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

