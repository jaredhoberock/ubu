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

template<view T, slicer_for<tensor_shape_t<T>> K>
class sliced_view : public view_base
{
  public:
    constexpr sliced_view(T tensor, K katana)
      : tensor_{tensor}, katana_{katana}
    {}

    using shape_type = slice_coordinate_result_t<tensor_shape_t<T>,K>;

    constexpr shape_type shape() const
    {
      return slice_coordinate(ubu::shape(tensor_), katana_);
    }

    template<congruent<shape_type> C>
    constexpr decltype(auto) operator[](const C& coord) const
    {
      auto tensor_coord = coordinate_cast<tensor_coordinate_t<T>>(unslice_coordinate(coord, katana_));
      return ubu::element(tensor_, tensor_coord);
    }

    template<congruent<shape_type> C>
    constexpr bool element_exists(const C& coord) const
    {
      auto tensor_coord = coordinate_cast<tensor_coordinate_t<T>>(unslice_coordinate(coord, katana_));
      return ubu::element_exists(tensor_, tensor_coord);
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

