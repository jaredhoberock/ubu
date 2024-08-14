#pragma once

#include "../../../detail/prologue.hpp"
#include "../../../utilities/constant.hpp"
#include "../../concepts/sized_tensor_like.hpp"
#include "../../concepts/view.hpp"
#include "../../coordinates/concepts/congruent.hpp"
#include "../../coordinates/coordinate_cast.hpp"
#include "../../coordinates/element.hpp"
#include "../../element_exists.hpp"
#include "../../shapes/shape.hpp"
#include "../../shapes/shape_size.hpp"
#include "../../traits/tensor_coordinate.hpp"
#include "../view_base.hpp"
#include "slice_coordinate.hpp"
#include "slicer.hpp"
#include "unslice_coordinate.hpp"
#include <utility>

namespace ubu
{

// XXX if K contains no underscore, this creates a view of
//     a single element; i.e. this view is a scalar with shape ()
//     however, the current coordinate & shape code doesn't
//     attempt to handle coordinates & shapes that are empty tuples
//     
//     instead, what we do here is introduce special cases which
//     promotes what would be a scalar view to a 1-rank tensor
//
//     ideally, we would would simply support 0-rank tensors
//     directly. once we do that, we can eliminate the is_singular
//     special cases in the code below
template<view T, slicer_for<tensor_shape_t<T>> K>
class sliced_view : public view_base
{
  private:
    // if slice_coordinate would return a (), that means this
    // is a singular view of exactly one element
    constexpr static bool is_singular = tuples::unit_like<slice_coordinate_result_t<tensor_shape_t<T>,K>>;

  public:
    constexpr sliced_view(T tensor, K katana)
      : tensor_{tensor}, katana_{katana}
    {}

    using shape_type = std::conditional_t<
      is_singular,
      constant<1>,
      slice_coordinate_result_t<tensor_shape_t<T>,K>
    >;

    constexpr shape_type shape() const
    {
      if constexpr (is_singular)
      {
        return 1_c;
      }
      else
      {
        return slice_coordinate(ubu::shape(tensor_), katana_);
      }
    }

    template<congruent<shape_type> C>
    constexpr decltype(auto) operator[](const C& coord) const
    {
      if constexpr (is_singular)
      {
        // when this is a singular view, the katana indicates the element of interest
        return ubu::element(tensor_, katana_);
      }
      else
      {
        // we have to do a coordinate_cast because unslice_coordinate may not return precisely tensor_coordinate_t
        // XXX ideally, this kind of cast would happen in a CPO for indexing a tensor with a coordinate

        auto tensor_coord = coordinate_cast<tensor_coordinate_t<T>>(unslice_coordinate(coord, katana_));
        return ubu::element(tensor_, tensor_coord);
      }
    }

    template<congruent<shape_type> C>
    constexpr bool element_exists(const C& coord) const
    {
      if constexpr (is_singular)
      {
        // when this is a singular view, the katana indicates the element of interest
        return ubu::element_exists(tensor_, katana_);
      }
      else 
      {
        auto tensor_coord = coordinate_cast<tensor_coordinate_t<T>>(unslice_coordinate(coord, katana_));
        return ubu::element_exists(tensor_, tensor_coord);
      }
    }

    // customize size if T is sized
    template<sized_tensor_like T_ = T>
    constexpr auto size() const
    {
      return shape_size(shape());
    }

  private:
    T tensor_;
    K katana_;
};

} // end ubu

#include "../../../detail/epilogue.hpp"

