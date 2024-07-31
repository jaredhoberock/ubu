#pragma once

#include "../detail/prologue.hpp"

#include "concepts/nested_tensor_like.hpp"
#include "coordinates/concepts/congruent.hpp"
#include "coordinates/coordinate_cat.hpp"
#include "coordinates/element.hpp"
#include "coordinates/split_coordinate_at.hpp"
#include "coordinates/traits/rank.hpp"
#include "element_exists.hpp"
#include "shapes/shape.hpp"
#include "traits/inner_tensor_shape.hpp"
#include "views/all.hpp"
#include "views/view_base.hpp"
#include <concepts>
#include <type_traits>

namespace ubu
{


template<nested_tensor_like P, congruent<inner_tensor_shape_t<P>> S = inner_tensor_shape_t<P>>
  requires std::is_object_v<P>
class quilted_tensor : public view_base_if<view<P>>
{
  private:
    static constexpr std::size_t inner_rank = rank_v<S>;

  public:
    template<nested_tensor_like OtherP>
      requires std::constructible_from<P,OtherP&&>
    constexpr quilted_tensor(OtherP&& patches, S patch_shape)
      : patches_(std::forward<OtherP>(patches)), patch_shape_(patch_shape)
    {}

    quilted_tensor(const quilted_tensor&) = default;

    using shape_type = coordinate_cat_t<S, tensor_shape_t<P>>;

    constexpr shape_type shape() const
    {
      return coordinate_cat(patch_shape_, ubu::shape(patches_));
    }

    template<congruent<shape_type> C>
    constexpr decltype(auto) operator[](C coord) const
    {
      auto [element_coord, patch_coord] = split_coordinate_at<inner_rank>(coord);
      return ubu::element(ubu::element(patches_, patch_coord), element_coord);
    }

    template<congruent<shape_type> C>
    constexpr bool element_exists(C coord) const
    {
      auto [element_coord, patch_coord] = split_coordinate_at<inner_rank>(coord);
      if(ubu::element_exists(patches_, patch_coord))
      {
        return ubu::element_exists(ubu::element(patches_, patch_coord), element_coord);
      }

      return false;
    }

    // we can customize slice when the katana slices one of the patches
    template<congruent<shape_type> K, class Pair = split_coordinate_at_t<inner_rank,K>>
      requires (slicer_for<tuples::first_t<Pair>,S> and congruent<tuples::second_t<Pair>,S>)
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

    // conditionally customize all if this quilted_tensor is not a view
    template<class P_ = P>
      requires (not view<P_>)
    constexpr view auto all() const
    {
      return make_quilted_tensor(ubu::all(patches_), patch_shape_);
    }

    template<class P_ = P>
      requires (not view<P_>)
    constexpr view auto all()
    {
      return make_quilted_tensor(ubu::all(patches_), patch_shape_);
    }

  private:
    template<nested_tensor_like OtherP>
    constexpr static quilted_tensor<OtherP,S> make_quilted_tensor(OtherP patches, S patch_shape)
    {
      return {patches, patch_shape};
    }

    P patches_;
    S patch_shape_;
};

template<nested_tensor_like P, congruent<inner_tensor_shape_t<P>> S>
quilted_tensor(P&&, S) -> quilted_tensor<std::remove_cvref_t<P>, S>;


} // end ubu

#include "../detail/epilogue.hpp"

