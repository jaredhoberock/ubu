#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../miscellaneous/tuples.hpp"
#include "../../coordinates.hpp"
#include "../../shapes/shape_size.hpp"
#include "../slices/dice_coordinate.hpp"
#include "../slices/slice_coordinate.hpp"
#include "../slices/slicer.hpp"
#include "detail/strided_layout_complement_impl.hpp"
#include "detail/strided_layout_compose_impl.hpp"
#include "layout.hpp"
#include "offset.hpp"
#include "strides/apply_stride.hpp"
#include "strides/apply_stride_r.hpp"
#include "strides/compact_left_major_stride.hpp"
#include "strides/stride_for.hpp"
#include <concepts>
#include <ranges>
#include <type_traits>


namespace ubu
{

template<coordinate S,
         stride_for<S> D = S,
         coordinate R = apply_stride_t<D,default_coordinate_t<S>>
        >
  requires congruent<R,apply_stride_t<D,default_coordinate_t<S>>>
class strided_layout : public std::ranges::view_base
{
  public:
    constexpr strided_layout(S shape, D stride)
      : shape_{shape}, stride_{stride}
    {}

    // this ctor is provided for CTAD. The coshape parameter is ignored otherwise.
    constexpr strided_layout(S shape, D stride, R /*coshape*/)
      : strided_layout{shape,stride}
    {}

    constexpr strided_layout(S shape)
      : strided_layout(shape, compact_left_major_stride(shape))
    {}

    strided_layout(const strided_layout&) = default;

    template<coordinate OtherS, stride_for<OtherS> OtherD>
      requires (std::convertible_to<OtherS,S> and std::convertible_to<OtherD,D>)
    constexpr strided_layout(const strided_layout<OtherS,OtherD>& other)
      : strided_layout{other.shape(), other.stride()}
    {}

    template<weakly_congruent<S> C>
    constexpr R operator[](const C& coord) const
    {
      return apply_stride_r<R>(stride(), colexicographical_lift(coord, shape()));
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

    constexpr R coshape() const
    {
      // this check avoids a divide by zero in operator[] that occurs
      // when one of the modes of shape is zero
      if(size() != 0)
      {
        R last_position = operator[](size() - 1);
        return coordinate_sum(last_position, ones_v<decltype(last_position)>);
      }
      else
      {
        return zeros_v<R>;
      }
    }

    // XXX the return type of this should be constrained to layout
    template<coordinate S1, stride_for<S1> D1, coordinate R1>
    constexpr auto compose(const strided_layout<S1,D1,R1>& other) const
    {
      auto [s,d] = detail::strided_layout_compose_impl(shape(), stride(), other.shape(), other.stride());
      return make_strided_layout_r<R>(s,d);
    }

    // XXX the return type of this should be constrained to layout
    template<coordinate... Ss, stride_for<Ss>... Ds, coordinate... Rs>
    constexpr auto concatenate(const strided_layout<Ss,Ds,Rs>&... layouts) const
    {
      // XXX what should be the resulting layout's result type?
      //     some concatenation of the Rs?
      
      using shape_tuple = std::conditional_t<tuples::tuple_like<S>, S, ubu::int1>;
      using stride_tuple = std::conditional_t<tuples::tuple_like<D>, D, ubu::int1>;

      return make_strided_layout(tuples::make_tuple_similar_to<shape_tuple>(shape(), layouts.shape()...),
                                 tuples::make_tuple_similar_to<stride_tuple>(stride(), layouts.stride()...));
    }

    // XXX the return type of this should be constrained to layout
    template<std::integral I>
    constexpr auto complement(I cosize_hi) const
    {
      // XXX what should be the resulting layout's result type? I?
      
      auto [s,d] = detail::strided_layout_complement_impl(shape(), stride(), cosize_hi);
      return make_strided_layout(s,d);
    }

    // XXX the return type of this should be constrained to layout
    constexpr auto complement() const
    {
      return complement(shape_size(coshape()));
    }

    template<slicer_for<S> K>
    constexpr layout auto slice(const K& katana) const
    {
      auto needs_offset = cute_slice(katana);
      auto diced_layout = cute_dice(katana);
      auto o = diced_layout[dice_coordinate(katana,katana)];
      return offset(needs_offset, o);
    }

  private:
    // XXX the return type of this is some type of strided_layout
    template<slicer_for<S> K>
    constexpr auto cute_slice(const K& katana) const
    {
      auto result_shape = slice_coordinate(shape(), katana);
      auto result_stride = slice_coordinate(stride(), katana);
      return make_strided_layout_r<R>(result_shape, result_stride);
    }

    // XXX the return type of this is some type of strided_layout
    template<slicer_for<S> K>
    constexpr auto cute_dice(const K& katana) const
    {
      auto result_shape = dice_coordinate(shape(), katana);
      auto result_stride = dice_coordinate(stride(), katana);
      return make_strided_layout_r<R>(result_shape, result_stride);
    }

    template<class S1, stride_for<S1> D1>
    constexpr static strided_layout<S1,D1> make_strided_layout(S1 s, D1 d) 
    {
      return {s, d};
    }

    template<class R1, class S1, stride_for<S1> D1>
      requires congruent<R,apply_stride_t<D1,S1>>
    constexpr static strided_layout<S1,D1,R1> make_strided_layout_r(S1 s, D1 d)
    {
      return {s, d};
    }

    S shape_;
    D stride_;
};


// When constructed from just a shape, we want the stride type to be the type
// returned by compact_left_major_stride(shape)
template<coordinate S>
strided_layout(S shape) -> strided_layout<S, compact_left_major_stride_t<S>>;


} // end ubu

#if __has_include(<fmt/format.h>)

// enable formatted output via fmtlib for ubu::strided_layout

#include <fmt/compile.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

template<ubu::coordinate S, ubu::stride_for<S> D>
struct fmt::formatter<ubu::strided_layout<S,D>>
{
  template<class ParseContext>
  constexpr auto parse(ParseContext& ctx)
  {
    return ctx.begin();
  }

  template<class FormatContext>
  constexpr auto format(const ubu::strided_layout<S,D>& l, FormatContext& ctx)
  {
    // using a compiled string allows formatting in device code
    return fmt::format_to(ctx.out(), FMT_COMPILE("{}:{}"), l.shape(), l.stride());
  }
};

// disable fmt detecting ubu::strided_layout as a range (in favor of the formatter above)
template<ubu::coordinate S, ubu::stride_for<S> D>
struct fmt::is_range<ubu::strided_layout<S,D>, char> : std::false_type {};

#endif // __has_include

#include "../../../detail/epilogue.hpp"

