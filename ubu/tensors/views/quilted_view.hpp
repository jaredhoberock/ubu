#pragma once

#include "../../detail/prologue.hpp"

#include "../concepts/nested_tensor_like.hpp"
#include "../coordinates/concepts/congruent.hpp"
#include "../coordinates/concepts/coordinate.hpp"
#include "../coordinates/coordinate_cat.hpp"
#include "../coordinates/element.hpp"
#include "../coordinates/split_coordinate_at.hpp"
#include "../coordinates/traits/rank.hpp"
#include "../element_exists.hpp"
#include "../shapes/shape.hpp"
#include "../traits/inner_tensor_shape.hpp"
#include "all.hpp"
#include "slices/slice.hpp"
#include "slices/slicer.hpp"
#include "view_base.hpp"
#include <concepts>
#include <type_traits>

namespace ubu
{


template<nested_tensor_like V, coordinate_for<V> S = inner_tensor_shape_t<V>>
  requires view<V>
class quilted_view : public view_base
{
  private:
    static constexpr std::size_t inner_rank = rank_v<S>;

  public:
    constexpr quilted_view(V patches, S patch_shape)
      : patches_(patches), patch_shape_(patch_shape)
    {}

    quilted_view(const quilted_view&) = default;

    using shape_type = coordinate_cat_result_t<S, tensor_shape_t<V>>;

    constexpr shape_type shape() const
    {
      return coordinate_cat(patch_shape_, ubu::shape(patches_));
    }

    template<congruent<shape_type> C>
    constexpr decltype(auto) operator[](C coord) const
    {
      auto leading_and_last = split_coordinate_at<inner_rank>(coord);
      return ubu::element(patches_, leading_and_last);
    }

    template<congruent<shape_type> C>
    constexpr bool element_exists(C coord) const
    {
      auto leading_and_last = split_coordinate_at<inner_rank>(coord);
      return ubu::element_exists(patches_, leading_and_last);
    }

    // we can customize slice when the katana slices one of the patches
    template<congruent<shape_type> K, class Pair = split_coordinate_at_result_t<inner_rank,K>>
      requires (slicer_for<tuples::first_t<Pair>,S> and coordinate_for<tuples::second_t<Pair>, V>)
    constexpr view auto slice(K katana) const
    {
      auto [patch_katana, patch_coord] = split_coordinate_at<inner_rank>(katana);
      return ubu::slice(ubu::element(patches_, patch_coord), patch_katana);
    }

    // quilt is the inverse of nestle, so our customization of nestle simply returns all(patches_)
    constexpr view auto nestle() const
    {
      return ubu::all(patches_);
    }

  private:
    V patches_;
    S patch_shape_;
};


} // end ubu

#include "../../detail/epilogue.hpp"

