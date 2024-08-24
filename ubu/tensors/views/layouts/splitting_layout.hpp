#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinates/concepts/congruent.hpp"
#include "../../coordinates/concepts/coordinate.hpp"
#include "../../coordinates/split_coordinate_at.hpp"
#include "../../coordinates/traits/rank.hpp"
#include "../../shapes/shape_size.hpp"
#include "../view_base.hpp"

namespace ubu
{

template<std::size_t split_point, coordinate S>
  requires (split_point < rank_v<S>)
class splitting_layout : public view_base
{
  public:
    constexpr splitting_layout(const S& shape)
      : shape_(shape)
    {}

    template<congruent<S> C>
    constexpr coordinate auto operator[](const C& coord) const
    {
      return split_coordinate_at<split_point>(coord);
    }

    constexpr S shape() const
    {
      return shape_;
    }

    constexpr coordinate auto coshape() const
    {
      return split_coordinate_at<split_point>(shape());
    }

    constexpr size_t size() const
    {
      return shape_size(shape());
    }

  private:
    S shape_;
};

} // end ubu

#include "../../../detail/epilogue.hpp"

