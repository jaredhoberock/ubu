#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/integrals/size.hpp"
#include "../concepts/composable.hpp"
#include "../concepts/sized_tensor.hpp"
#include "../concepts/tensor.hpp"
#include "../concepts/view.hpp"
#include "../coordinates/element.hpp"
#include "../element_exists.hpp"
#include "../shapes/shape.hpp"
#include "../traits/tensor_coordinate.hpp"
#include "../traits/tensor_shape.hpp"
#include "../vectors/span_like.hpp"
#include "layouts/concepts/coshaped_layout.hpp"
#include "layouts/coshape.hpp"
#include "all.hpp"
#include "view_base.hpp"


namespace ubu
{


template<class A, view B>
  requires (std::is_trivially_copy_constructible_v<A> and composable<A,B> and (view<A> or not tensor<A>))
class composed_view : public view_base
{
  public:
    using shape_type = tensor_shape_t<B>;
    using coordinate_type = tensor_coordinate_t<B>;

    constexpr composed_view(A a, B b)
      : a_(a), b_(b)
    {}

    composed_view(const composed_view&) = default;

    constexpr shape_type shape() const
    {
      return ubu::shape(b_);
    }

    template<coshaped_layout A_ = A>
    constexpr auto coshape() const
    {
      return ubu::coshape(a());
    }

    template<class A_ = A, class B_ = B>
      requires (not tensor<A_> and sized_tensor<B_>)
    constexpr auto size() const
    {
      return ubu::size(b_);
    }

    // precondition: element_exists(coord)
    template<coordinate_for<B> C>
    constexpr decltype(auto) operator[](const C& coord) const
    {
      return element(a_, element(b_,coord));
    }

    template<coordinate_for<B> C>
    constexpr bool element_exists(const C& coord) const
    {
      if (not ubu::element_exists(b_, coord)) return false;

      // avoid evaluating element(b_, coord) if we don't need to
      if constexpr (shaped<A>)
      {
        auto a_coord = element(b_, coord);

        return ubu::element_exists(a_, a_coord);
      }

      return true;
    }

    constexpr A a() const
    {
      return a_;
    }

    constexpr B b() const
    {
      return b_;
    }

    // returns the pair (a(),b())
    constexpr std::pair<A,B> decompose() const
    {
      return {a(), b()};
    }

    // returns a() if it is span_like
    // XXX TODO eliminate this function
    template<span_like S = A>
    constexpr S span() const
    {
      return a();
    }

  private:
    A a_;
    B b_;
};


} // end ubu

#include "../../detail/epilogue.hpp"

