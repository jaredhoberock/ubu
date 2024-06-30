#pragma once

#include "../../detail/prologue.hpp"

#include "../concepts/tensor_like.hpp"
#include "../concepts/view.hpp"
#include "../coordinates/concepts/congruent.hpp"
#include "../coordinates/traits/rank.hpp"
#include "../detail/coordinate_cat.hpp"
#include "../detail/coordinate_tail.hpp"
#include "../shapes/shape.hpp"
#include "../traits/tensor_shape.hpp"
#include "../traits/tensor_rank.hpp"
#include "all.hpp"
#include "slices/underscore.hpp"
#include <ranges>
#include <utility>

namespace ubu
{


template<view T>
  requires (tensor_rank_v<T> > 1)
class nestled_view : public std::ranges::view_base
{
  public:
    constexpr nestled_view(T tensor)
      : tensor_{tensor}
    {}

    nestled_view(const nestled_view&) = default;

    using shape_type = decltype(detail::coordinate_tail(std::declval<tensor_shape_t<T>>()));

    constexpr shape_type shape() const
    {
      // return the tail elements of the tensor's shape
      return detail::coordinate_tail(ubu::shape(tensor_));
    }

    template<congruent<shape_type> C>
    constexpr auto operator[](const C& coord) const
    {
      return slice(tensor_, detail::coordinate_cat(ubu::_, coord));
    }

    constexpr std::size_t size() const
    {
      return shape_size(shape());
    }

  private:
    T tensor_;
};


} // end ubu

#include "../../detail/epilogue.hpp"

