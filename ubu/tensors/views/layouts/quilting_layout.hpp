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

// XXX this is called a "quilting" layout because it does
//     exactly what was needed by the old quilted_view
//     it still feels a little janky, like we need some more
//     primitive concatenating_layout type and then this layout
//     could adapt that in some simple way
template<layout L, layout R>
  requires (view<L> and view<R>)
class quilting_layout : public view_base
{
  private:
    constexpr static std::size_t split_position = rank_v<shape_t<L>>;

  public:
    constexpr quilting_layout(L left_layout, R right_layout)
      : left_layout_(left_layout), right_layout_(right_layout)
    {}

    template<coordinate LS, coordinate RS>
      requires (std::constructible_from<L,LS> and std::constructible_from<R,RS>)
    constexpr quilting_layout(LS left_shape, RS right_shape)
      : quilting_layout(L(left_shape), R(right_shape))
    {}

    quilting_layout(const quilting_layout&) = default;

    using shape_type = coordinate_cat_result_t<shape_t<L>,shape_t<R>>;

    constexpr shape_type shape() const
    {
      return coordinate_cat(ubu::shape(left_layout_), ubu::shape(right_layout_));
    }

    constexpr coordinate auto operator[](congruent<shape_type> auto coord) const
    {
      auto [left_coord, right_coord] = split_coordinate_at<split_position>(coord);

      return std::pair(ubu::element(left_layout_, left_coord), ubu::element(right_layout_, right_coord));
    }

    constexpr bool element_exists(congruent<shape_type> auto coord) const
    {
      auto [left_coord, right_coord] = split_coordinate_at<split_position>(coord);
      return ubu::element_exists(left_layout_, left_coord) and ubu::element_exists(right_layout_, right_coord);
    }

    // customize size if both L and R are sized
    template<sized_tensor_like L_ = L, sized_tensor_like R_ = R>
    constexpr integral_like auto size() const
    {
      return shape_size(shape());
    }

    // customize coshape if both L and R are coshaped
    template<coshaped_layout L_ = L, coshaped_layout R_ = R>
    constexpr coordinate auto coshape() const
    {
      return std::pair(ubu::coshape(left_layout_), ubu::coshape(right_layout_));
    }

    // customize slice if we can split the katana and use the two pieces to slice both layouts
    // XXX i think we could eliminate these constraints if ubu::slice could handle singular slices
    template<equal_rank<shape_type> K>
      requires (    sliceable<L, tuples::first_t<split_coordinate_at_result_t<split_position,K>>>
                and sliceable<R, tuples::second_t<split_coordinate_at_result_t<split_position,K>>>)
    constexpr layout auto slice(K katana) const
    {
      auto [left_katana, right_katana] = split_coordinate_at<split_position>(katana);

      // XXX this should ideally return the result of a concatenate CPO
      return make_quilting_layout(ubu::slice(left_layout_, left_katana), ubu::slice(right_layout_, right_katana));
    }

  private:
    template<layout OtherL, layout OtherR>
      requires (view<OtherL> and view<OtherR>)
    constexpr static quilting_layout<OtherL,OtherR> make_quilting_layout(OtherL l, OtherR r)
    {
      return {l,r};
    }

    L left_layout_;
    R right_layout_;
};

template<coordinate L, coordinate R>
quilting_layout(L,R) -> quilting_layout<identity_layout<L>, identity_layout<R>>;

} // end ubu

#include "../../../detail/epilogue.hpp"

