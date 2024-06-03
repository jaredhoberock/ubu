#pragma once

#include "../detail/prologue.hpp"

#include "../miscellaneous/constant.hpp"
#include "../miscellaneous/integral/size.hpp"
#include "coordinate/concepts/congruent.hpp"
#include "coordinate/coordinate_cast.hpp"
#include "coordinate/detail/tuple_algorithm.hpp"
#include "coordinate/element.hpp"
#include "concepts/tensor_like.hpp"
#include "concepts/same_tensor_rank.hpp"
#include "concepts/sized_tensor_like.hpp"
#include "concepts/view.hpp"
#include "detail/stacked_shape.hpp"
#include "detail/subtract_element.hpp"
#include "domain.hpp"
#include "element_exists.hpp"
#include "layout/layout.hpp"
#include "shape/shape.hpp"
#include "slice/slice.hpp"
#include "slice/slicer.hpp"
#include "traits/tensor_reference.hpp"
#include "views/compose.hpp"
#include <ranges>
#include <span>
#include <type_traits>


namespace ubu
{


template<std::size_t axis_, view A, view B, class R = std::common_type_t<tensor_reference_t<A>, tensor_reference_t<B>>>
  requires (axis_ <= tensor_rank_v<A> and same_tensor_rank<A,B>)
class stacked_view;


template<std::size_t axis, tensor_like A, same_tensor_rank<A> B>
  requires (axis <= tensor_rank_v<A>)
constexpr stacked_view<axis, all_t<A&&>, all_t<B&&>> stack(A&& a, B&& b);


// presents of a single view of two tensors stacked along the given axis
template<std::size_t axis_, view A, view B, class R>
  requires (axis_ <= tensor_rank_v<A> and same_tensor_rank<A,B>)
class stacked_view : public std::ranges::view_base
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
      return ubu::size(a_) + ubu::size(b_);
    }

    template<slicer_for<shape_type> K, class A_ = A, class B_ = B>
      requires std::same_as<A_,B_>
    constexpr tensor_like auto slice(const K& katana) const
    {
      // when A and B are the same type, and katana does not cut the stacking axis,
      // (i.e., the slice is completely contained within A or B)
      // then we can customize slice to yield a result simpler than the result 
      // of the generic version of slice
      // XXX in cases where katana does cut the stacking axis, then the
      //     result would be a stack of slices. in such a case, A and B need not be
      //     the same type. we could customize slice for that case as well
    
      // XXX this assumes that katana does not cut the stacking axis
      auto [stratum, local_katana] = to_stratum_and_local_slicer(katana);
      return (stratum == 0) ? ubu::slice(a(), local_katana) : ubu::slice(b(), local_katana);
    }

    template<class T, class Self = stacked_view>
      requires layout_for<Self, std::span<T>>
    friend view auto compose(const std::span<T>& s, const stacked_view& self)
    {
      return stack<axis_>(ubu::compose(s, self.a()), ubu::compose(s, self.b()));
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
          // local_coord needs to have the same type as coord, so cast the result of subtract_element
          C local_coord = coordinate_cast<C>(detail::subtract_element<axis>(coord, bound));
          return std::pair(1, local_coord);
        }
      }
      else
      {
        // when axis == tensor_rank_v<A>,
        // the final mode of the shape is 2
        // so the final mode of coord selects the stratum, either a or b
        auto stratum = element(coord, axis);
        auto local_coord = detail::tuple_drop_last_and_unwrap_single(coord);
        return std::pair(stratum, local_coord);
      }
    }

    // returns the pair (stratum, local_katana)
    // XXX this only makes sense if the axis'th element of K contains no underscore
    template<slicer_for<shape_type> K, class A_ = A, class B_ = B>
      requires coordinate<std::tuple_element_t<axis_,K>>
    constexpr detail::pair_like auto to_stratum_and_local_slicer(const K& katana) const
    {
      // there is no underscore in the axis'th mode of K,
      // which means that our slice is contained within either A or B

      auto bound = element(ubu::shape(a_), axis);

      // check if katana[axis] < shape(a)[axis]
      if(is_below(element(katana, axis), bound))
      {
        return std::pair(0, katana);
      }
      else
      {
        // XXX we probably need to do a cast to K
        K local_katana = detail::subtract_element<axis>(katana, bound);
        return std::pair(1, local_katana);
      }
    }
};


template<std::size_t axis, tensor_like A, same_tensor_rank<A> B>
  requires (axis <= tensor_rank_v<A>)
constexpr stacked_view<axis, all_t<A&&>, all_t<B&&>> stack(A&& a, B&& b)
{
  return {all(std::forward<A>(a)), all(std::forward<B>(b))};
}


} // end ubu

#include "../detail/epilogue.hpp"

