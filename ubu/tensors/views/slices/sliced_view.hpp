#pragma once

#include "../../../detail/prologue.hpp"
#include "../../concepts/sized_tensor_like.hpp"
#include "../../concepts/view.hpp"
#include "../../coordinates/coordinate_cast.hpp"
#include "../../element_exists.hpp"
#include "../../shapes/shape.hpp"
#include "../../shapes/shape_size.hpp"
#include "../../traits/tensor_coordinate.hpp"
#include "slice_coordinate.hpp"
#include "slicer.hpp"
#include "unslice_coordinate.hpp"
#include <ranges>
#include <utility>

namespace ubu
{

template<view T, slicer_for<tensor_shape_t<T>> S>
class sliced_view : public std::ranges::view_base
{
  public:
    constexpr sliced_view(T tensor, S katana)
      : tensor_{tensor}, katana_{katana}
    {}

    using shape_type = decltype(slice_coordinate(ubu::shape(std::declval<T>()), std::declval<S>()));

    constexpr shape_type shape() const
    {
      return slice_coordinate(ubu::shape(tensor_), katana_);
    }

    constexpr decltype(auto) operator[](shape_type coord) const
    {
      // we have to do a coordinate_cast because unsliced_coord may not return precisely tensor_coordinate_t
      // XXX ideally, this kind of cast would happen in a CPO for indexing a tensor with a coordinate

      auto tensor_coord = coordinate_cast<tensor_coordinate_t<T>>(unslice_coordinate(coord, katana_));
      return tensor_[tensor_coord];
    }

    template<class T_ = T>
      requires sized_tensor_like<T_>
    constexpr bool element_exists(shape_type coord) const
    {
      auto tensor_coord = coordinate_cast<tensor_coordinate_t<T>>(unslice_coordinate(coord, katana_));
      return ubu::element_exists(tensor_, tensor_coord);
    }

    // if T is sized, we can provide size
    template<class T_ = T>
      requires sized_tensor_like<T_>
    constexpr bool size() const
    {
      return shape_size(shape());
    }

  private:
    T tensor_;
    S katana_;
};

} // end ubu

#include "../../../detail/epilogue.hpp"

