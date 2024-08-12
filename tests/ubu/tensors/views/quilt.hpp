#include <cassert>
#include <ubu/tensors/iterators.hpp>
#include <ubu/tensors/traits/tensor_reference.hpp>
#include <ubu/tensors/views/all.hpp>
#include <ubu/tensors/views/slices.hpp>
#include <ubu/tensors/views/quilt.hpp>
#include <concepts>
#include <ranges>
#include <vector>

void test_quilt()
{
  using namespace ubu;

  std::vector<std::vector<int>> nested_vec{{13,7}, {42,66}};
  auto quilted = quilt(nested_vec, 2);

  {
    // test element_exists & element

    assert(element_exists(quilted, std::pair(0,0)));
    assert(element_exists(quilted, std::pair(1,0)));
    assert(element_exists(quilted, std::pair(0,1)));
    assert(element_exists(quilted, std::pair(1,1)));

    assert(not element_exists(quilted, std::pair(0,2)));
    assert(not element_exists(quilted, std::pair(1,2)));
    assert(not element_exists(quilted, std::pair(2,0)));
    assert(not element_exists(quilted, std::pair(2,1)));

    assert(13 == quilted[std::pair(0,0)]);
    assert( 7 == quilted[std::pair(1,0)]);
    assert(42 == quilted[std::pair(0,1)]);
    assert(66 == quilted[std::pair(1,1)]);

    std::vector<int> expected{{13,7,42,66}};
    assert(std::ranges::equal(expected, quilted));
  }

  {
    // test slice

    {
      // slice across patch boundaries

      auto row_0 = slice(quilted, std::pair(0, _));
      std::vector<int> expected_row_0({13,42});
      assert(std::ranges::equal(expected_row_0, row_0));

      auto row_1 = slice(quilted, std::pair(1, _));
      std::vector<int> expected_row_1({7,66});
      assert(std::ranges::equal(expected_row_1, row_1));
    }


    {
      // slice along patch boundaries
      using expected_slice_type = all_t<tensor_reference_t<decltype(nested_vec)>>;

      auto patch_0 = slice(quilted, std::pair(_,0));
      std::vector<int> expected_patch_0({13,7});

      static_assert(std::same_as<expected_slice_type, decltype(patch_0)>);
      assert(std::ranges::equal(expected_patch_0, patch_0));

      auto patch_1 = slice(quilted, std::pair(_,1));
      std::vector<int> expected_patch_1({42,66});

      static_assert(std::same_as<expected_slice_type, decltype(patch_1)>);
      assert(std::ranges::equal(expected_patch_1, patch_1));
    }
  }
}

