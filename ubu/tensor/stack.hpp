#pragma once

#include "../detail/prologue.hpp"

#include "coordinate/concepts/congruent.hpp"
#include "coordinate/constant.hpp"
#include "coordinate/detail/tuple_algorithm.hpp"
#include "coordinate/element.hpp"
#include "concepts/tensor_like.hpp"
#include "concepts/same_tensor_rank.hpp"
#include "concepts/sized_tensor_like.hpp"
#include "detail/stacked_shape.hpp"
#include "detail/subtract_element.hpp"
#include "element_exists.hpp"
#include "domain.hpp"
#include "shape/shape.hpp"
#include "traits/tensor_reference.hpp"
#include <ranges>
#include <type_traits>


namespace ubu
{


// presents of a single view of two tensors stacked along the given axis
template<std::size_t axis_, tensor_like A, same_tensor_rank<A> B, class R = std::common_type_t<tensor_reference_t<A>, tensor_reference_t<B>>>
  requires (axis_ <= tensor_rank_v<A>)
class stacked_view
{
  public:
    static constexpr constant<axis_> axis;

    constexpr stacked_view(A a, B b)
      : a_{a}, b_{b}
    {}

    stacked_view(const stacked_view&) = default;

    constexpr A a() const
    {
      return a_;
    }

    constexpr B b() const
    {
      return b_;
    }

    using shape_type = decltype(detail::stacked_shape<axis_>(ubu::shape(std::declval<A>()), ubu::shape(std::declval<B>())));

    constexpr shape_type shape() const
    {
      return detail::stacked_shape<axis_>(ubu::shape(a_), ubu::shape(b_));
    }

    template<congruent<shape_type> C>
    constexpr bool element_exists(const C& coord) const
    {
      auto [stratum, local_coord] = to_stratum_and_local_coord(coord);

      if(stratum == 0)
      {
        return in_domain(a_, local_coord) and ubu::element_exists(a_, local_coord);
      }

      return in_domain(b_, local_coord) and ubu::element_exists(b_, local_coord);
    }

    template<congruent<shape_type> C>
    constexpr R operator[](const C& coord) const
    {
      auto [stratum, local_coord] = to_stratum_and_local_coord(coord);
      return stratum == 0 ? element(a_, local_coord) : element(b_, local_coord);
    }

    template<class A_ = A, class B_ = B>
      requires (sized_tensor_like<A_> and sized_tensor_like<B_>)
    constexpr auto size() const
    {
      return std::ranges::size(a_) + std::ranges::size(b_);
    }

  private:
    A a_;
    B b_;

    // returns the pair (stratum, local_coord)
    template<congruent<shape_type> C>
    constexpr detail::pair_like auto to_stratum_and_local_coord(const C& coord) const
    {
      using namespace ubu;

      if constexpr (axis < tensor_rank_v<A>)
      {
        // when axis < tensor_rank_v<A>,
        // to select between a and b,
        // we check the axis'th mode of coord against
        // the corresponding mode in shape(a)

        auto bound = element(ubu::shape(a_), axis);

        // check if coord[axis] < shape(a)[axis]
        if(is_below(element(coord, axis), bound))
        {
          return std::pair(0, coord);
        }
        else
        {
          auto local_coord = detail::subtract_element<axis>(coord, bound);
          return std::pair(1, local_coord);
        }
      }
      else
      {
        // when axis == tensor_rank_v<A>,
        // the final mode of the shape is 2
        // so the final mode of coord selects the straum, either a or b
        auto stratum = element(coord, axis);
        auto local_coord = detail::tuple_drop_last_and_unwrap_single(coord);
        return std::pair(stratum, local_coord);
      }
    }
};


template<std::size_t axis, tensor_like A, same_tensor_rank<A> B>
  requires (axis <= tensor_rank_v<A>)
constexpr stacked_view<axis,A,B> stack(A a, B b)
{
  return {a,b};
}


} // end ubu

#include "../detail/epilogue.hpp"

