#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinates/concepts/coordinate.hpp"
#include "../../coordinates/concepts/equal_rank.hpp"
#include "../../coordinates/coordinate_cat.hpp"
#include "../../coordinates/split_coordinate_at.hpp"
#include "../../coordinates/traits/rank.hpp"
#include "../../shapes/shape.hpp"
#include "../../shapes/shape_size.hpp"
#include "../slices/slice.hpp"
#include "../slices/sliceable.hpp"
#include "../slices/slicer.hpp"
#include "../view_base.hpp"
#include "concepts/coshaped_layout.hpp"
#include "concepts/layout.hpp"
#include "coshape.hpp"
#include "identity_layout.hpp"

namespace ubu
{

template<layout L, layout R>
  requires (view<L> and view<R>)
class concatenating_layout : public view_base
{
  private:
    constexpr static std::size_t split_position = rank_v<shape_t<L>>;

  public:
    constexpr concatenating_layout(L left_layout, R right_layout)
      : left_layout_(left_layout), right_layout_(right_layout)
    {}

    template<coordinate LS, coordinate RS>
      requires (std::constructible_from<L,LS> and std::constructible_from<R,RS>)
    constexpr concatenating_layout(LS left_shape, RS right_shape)
      : concatenating_layout(L(left_shape), R(right_shape))
    {}

    concatenating_layout(const concatenating_layout&) = default;

    using shape_type = coordinate_cat_result_t<shape_t<L>,shape_t<R>>;

    constexpr shape_type shape() const
    {
      return coordinate_cat(ubu::shape(left_layout_), ubu::shape(right_layout_));
    }

    // provide size if both L and R are sized
    template<sized_tensor_like L_ = L, sized_tensor_like R_ = R>
    constexpr integral_like auto size() const
    {
      return shape_size(shape());
    }

    // provide coshape if both L and R are coshaped
    template<coshaped_layout L_ = L, coshaped_layout R_ = R>
    constexpr coordinate auto coshape() const
    {
      return coordinate_cat(ubu::coshape(left_layout_), ubu::coshape(right_layout_));
    }

    constexpr auto operator[](congruent<shape_type> auto coord) const
    {
      auto [left_coord, right_coord] = split_coordinate_at<split_position>(coord);
      return std::pair(ubu::element(left_layout_, left_coord), ubu::element(right_layout_, right_coord));
    }

    constexpr bool element_exists(congruent<shape_type> auto coord) const
    {
      auto [left_coord, right_coord] = split_coordinate_at<split_position>(coord);
      return ubu::element_exists(left_layout_, left_coord) and ubu::element_exists(right_layout_, right_coord);
    }

    // we customize slice if we can split the katana and use the two pieces to slice both layouts
    template<equal_rank<shape_type> K>
      requires (    sliceable<L, tuples::first_t<split_coordinate_at_result_t<split_position,K>>>
                and sliceable<R, tuples::second_t<split_coordinate_at_result_t<split_position,K>>>)
    constexpr layout auto slice(K katana) const
    {
      auto [left_katana, right_katana] = split_coordinate_at<split_position>(katana);

      // XXX this should ideally return the result of a concatenate CPO
      return make_concatenating_layout(ubu::slice(left_layout_, left_katana), ubu::slice(right_layout_, right_katana));
    }

  private:
    template<layout OtherL, layout OtherR>
      requires (view<OtherL> and view<OtherR>)
    constexpr static concatenating_layout<OtherL,OtherR> make_concatenating_layout(OtherL l, OtherR r)
    {
      return {l,r};
    }

    L left_layout_;
    R right_layout_;
};

template<coordinate L, coordinate R>
concatenating_layout(L,R) -> concatenating_layout<identity_layout<L>, identity_layout<R>>;

} // end ubu

#include "../../../detail/epilogue.hpp"

