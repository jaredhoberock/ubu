#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate.hpp"
#include "../shape/shape_size.hpp"
#include "../slice/dice_coordinate.hpp"
#include "../slice/slice_coordinate.hpp"
#include "../slice/slicer.hpp"
#include "detail/strided_layout_complement_impl.hpp"
#include "detail/strided_layout_compose_impl.hpp"
#include "layout.hpp"
#include "stride/apply_stride.hpp"
#include "stride/compact_column_major_stride.hpp"
#include "stride/stride_for.hpp"
#include <concepts>


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
      : strided_layout(s, compact_column_major_stride(s))
    {}

    strided_layout(const strided_layout&) = default;

    template<coordinate OtherS, stride_for<OtherS> OtherD>
      requires (std::convertible_to<OtherS,S> and std::convertible_to<OtherD,D>)
    constexpr strided_layout(const strided_layout<OtherS,OtherD>& other)
      : strided_layout{other.shape(), other.stride()}
    {}

    template<weakly_congruent<S> C>
    constexpr coordinate auto apply_layout(const C& coord) const
    {
      return apply_stride(colexicographical_lift(coord, shape()), stride());
    }

    template<weakly_congruent<S> C>
    constexpr coordinate auto operator[](const C& coord) const
    {
      return apply_layout(coord);
    }

    constexpr S shape() const
    {
      return shape_;
    }

    constexpr auto size() const
    {
      return shape_size(shape());
    }

    constexpr D stride() const
    {
      return stride_;
    }

    // XXX consider whether the following functions need to be members

    // XXX the return type of this should be the same as operator[]
    constexpr coordinate auto coshape() const
    {
      auto last_position = apply_layout(size() - 1);
      return coordinate_sum(last_position, ones<decltype(last_position)>);
    }

    // XXX the return type of this should be constrained to layout
    template<coordinate S1, stride_for<S1> D1>
    constexpr auto compose(const strided_layout<S1,D1>& other) const
    {
      auto [s,d] = detail::strided_layout_compose_impl(shape(), stride(), other.shape(), other.stride());
      return make_strided_layout(s,d);
    }

    // XXX the return type of this should be constrained to layout
    template<coordinate... Ss, stride_for<Ss>... Ds>
    constexpr auto concatenate(const strided_layout<Ss,Ds>&... layouts) const
    {
      using shape_tuple = std::conditional_t<detail::tuple_like<S>, S, ubu::int1>;
      using stride_tuple = std::conditional_t<detail::tuple_like<D>, D, ubu::int1>;

      return make_strided_layout(detail::make_tuple_similar_to<shape_tuple>(shape(), layouts.shape()...),
                                 detail::make_tuple_similar_to<stride_tuple>(stride(), layouts.stride()...));
    }

    // XXX the return type of this should be constrained to layout
    template<std::integral I>
    constexpr auto complement(I cosize_hi) const
    {
      auto [s,d] = detail::strided_layout_complement_impl(shape(), stride(), cosize_hi);
      return make_strided_layout(s,d);
    }

    // XXX the return type of this should be constrained to layout
    constexpr auto complement() const
    {
      return complement(shape_size(coshape()));
    }

    // XXX the return type of this is some type of strided_layout
    template<slicer_for<S> K>
    constexpr auto slice(const K& katana) const
    {
      auto result_shape = slice_coordinate(shape(), katana);
      auto result_stride = slice_coordinate(stride(), katana);
      return make_strided_layout(result_shape, result_stride);
    }

    // XXX the return type of this is some type of strided_layout
    template<slicer_for<S> K>
    constexpr auto dice(const K& katana) const
    {
      auto result_shape = dice_coordinate(shape(), katana);
      auto result_stride = dice_coordinate(stride(), katana);
      return make_strided_layout(result_shape, result_stride);
    }

  private:
    template<class S1, stride_for<S1> D1>
    constexpr static strided_layout<S1,D1> make_strided_layout(S1 s, D1 d) 
    {
      return {s, d};
    }

    S shape_;
    D stride_;
};


} // end ubu

#if __has_include(<fmt/format.h>)

#include <fmt/format.h>

template<ubu::coordinate S, ubu::stride_for<S> D>
struct fmt::formatter<ubu::strided_layout<S,D>>
{
  template<class ParseContext>
  constexpr auto parse(ParseContext& ctx)
  {
    return ctx.begin();
  }

  template<class FormatContext>
  auto format(const ubu::strided_layout<S,D>& l, FormatContext& ctx)
  {
    return fmt::format_to(ctx.out(), "{}:{}", l.shape(), l.stride());
  }
};

#endif // __has_include

#include "../../detail/epilogue.hpp"

