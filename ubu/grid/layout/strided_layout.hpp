#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate.hpp"
#include "stride/apply_stride.hpp"
#include "stride/compact_stride.hpp"
#include "stride/stride_for.hpp"


namespace ubu
{


template<coordinate S, stride_for<S> D = S>
class strided_layout
{
  public:
    constexpr strided_layout(S s, D d)
      : shape_{s}, stride_{d}
    {}

    constexpr strided_layout(S s)
      : strided_layout(s, compact_stride(s))
    {}

    strided_layout(const strided_layout&) = default;

    template<weakly_congruent<S> C>
    constexpr auto operator()(C coord) const
    {
      return apply_stride(congrue_coordinate(coord, shape()), stride_);
    }

    constexpr S shape() const
    {
      return shape_;
    }

    constexpr D stride() const
    {
      return stride_;
    }

  private:
    S shape_;
    D stride_;
};


} // end ubu

#include "../../detail/epilogue.hpp"

