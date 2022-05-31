#include <algorithm>
#include <ubu/memory/fancy_ptr.hpp>

#undef NDEBUG
#include <cassert>

namespace ns = ubu;


template<class T>
struct trivial_copier
{
  using handle_type = T*;
  using element_type = T;
  using value_type = std::remove_cv_t<element_type>;

  static value_type* copy_n_to_raw_pointer(handle_type from, std::size_t n, value_type* to)
  {
    return std::copy_n(from, n, to);
  }

  static handle_type copy_n_from_raw_pointer(const value_type* from, std::size_t n, handle_type to)
  {
    return std::copy_n(from, n, to);
  }

  static handle_type copy_n(handle_type from, std::size_t n, handle_type to)
  {
    return std::copy_n(from, n, to);
  }
};


void test_fancy_ptr()
{
  using namespace ns;

  {
    // test concepts
    static_assert(std::random_access_iterator<fancy_ptr<int, trivial_copier<int>>>);
  }

  {
    // test default construction
    fancy_ptr<int, trivial_copier<int>> ptr;

    // silence "declared but never referenced" warnings
    static_cast<void>(ptr);
  }

  {
    // test construction from nullptr
    fancy_ptr<int, trivial_copier<int>> ptr{nullptr};

    assert(ptr.get() == nullptr);
    assert(!ptr);
  }

  {
    int array[] = {0, 1, 2, 3};

    // test construction from raw pointer
    fancy_ptr<int, trivial_copier<int>> ptr(array);

    // test native_handle
    for(int i = 0; i < 4; ++i)
    {
      assert((ptr + i).native_handle() == &array[i]);
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
    fancy_ptr<const int, trivial_copier<const int>> ptr(array);

    // test native_handle
    for(int i = 0; i < 4; ++i)
    {
      assert((ptr + i).native_handle() == &array[i]);
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

