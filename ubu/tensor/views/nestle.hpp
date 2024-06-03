#pragma once

#include "../../detail/prologue.hpp"

#include "../concepts/tensor_like.hpp"
#include "../concepts/view.hpp"
#include "../coordinate/concepts/congruent.hpp"
#include "../coordinate/detail/tuple_algorithm.hpp"
#include "../coordinate/traits/rank.hpp"
#include "../detail/coordinate_cat.hpp"
#include "../detail/coordinate_tail.hpp"
#include "../shape/shape.hpp"
#include "../slice/underscore.hpp"
#include "../traits/tensor_shape.hpp"
#include "../traits/tensor_rank.hpp"
#include "all.hpp"
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


// XXX in general we would like to be able to nestle up to rank-1 leading dims
template<tensor_like T>
  requires (tensor_rank_v<T> > 1)
constexpr auto nestle(T&& tensor)
{
  return nestled_view(all(std::forward<T>(tensor)));
}


} // end ubu

#include "../../detail/epilogue.hpp"

