#pragma once

#include "../detail/prologue.hpp"

#include "concepts/tensor_like.hpp"
#include "slice/underscore.hpp"
#include "traits/tensor_shape.hpp"
#include "traits/tensor_rank.hpp"
#include "coordinate/concepts/congruent.hpp"
#include "coordinate/detail/tuple_algorithm.hpp"
#include "coordinate/traits/rank.hpp"
#include "detail/coordinate_cat.hpp"
#include "detail/coordinate_tail.hpp"
#include <utility>

namespace ubu
{


template<tensor_like T>
  requires (tensor_rank_v<T> > 1)
class nestled_view
{
  public:
    constexpr nestled_view(const T& tensor)
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


// XXX in general we would like to be able to nestle up to rank-1 leading dims
template<tensor_like T>
  requires (tensor_rank_v<T> > 1)
constexpr nestled_view<T> nestle(const T& tensor)
{
  return {tensor};
}


} // end ubu

#include "../detail/epilogue.hpp"

