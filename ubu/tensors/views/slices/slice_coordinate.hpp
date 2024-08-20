#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/tuples.hpp"
#include "../../coordinates/concepts/coordinate.hpp"
#include "slicer.hpp"
#include "underscore.hpp"
#include <tuple>
#include <utility>


namespace ubu
{


template<slicer C, slicer_for<C> K>
constexpr semicoordinate auto slice_coordinate(const C& coord, const K& katana)
{
  using namespace ubu;

  if constexpr (slicer_without_underscore<K>)
  {
    // terminal case 0: katana contains no underscore, discard the whole coordinate
    return std::tuple();
  }
  else if constexpr (detail::is_underscore_v<K>)
  {
    // terminal case 1: katana is literally the underscore, keep the whole coordinate
    return coord;
  }
  else
  {
    static_assert(tuples::same_size<C,K>);

    // recursive case: the katana contains at least one underscore and is a tuple

    auto coord_and_katana = tuples::zip(coord,katana);

    // the idea is to apply slice_coordinate(c_i,k_i) and concatenate all the results
    auto result_tuple = tuples::fold_left(coord_and_katana, std::tuple(), [](const auto& prev_result, const auto& c_i_and_k_i)
    {
      const auto& [c_i, k_i] = c_i_and_k_i;

      auto r_i = slice_coordinate(c_i,k_i);

      // before concatenating r_i, we may need to wrap it in an extra tuple layer
      // for the concatenation to work out. We do this if:
      //
      // 1. r_i is not a tuple, or
      // 2. k_i is _, or
      // 3. k_i contains an underscore and r is (), in this case a () element was selected by k_i
      //
      // Case 2. preserves the tuple structure of any tuples selected by k_i

      auto maybe_wrap = [&]<class T>(const T& r)
      {
        if constexpr (not tuples::tuple_like<T> or
                      detail::is_underscore_v<decltype(k_i)> or
                      (slicer_with_underscore<decltype(k_i)> and tuples::unit_like<T>))
        {
          return tuples::make_like<C>(r);
        }
        else
        {
          return r;
        }
      };

      return tuples::concatenate_like<C>(prev_result, maybe_wrap(r_i));
    });

    // finally, because it's really inconvenient for this function to return a (single_thing), we unwrap any singles we find
    return tuples::unwrap_single(result_tuple);
  }
}


template<slicer C, slicer_for<C> K>
using slice_coordinate_result_t = decltype(slice_coordinate(std::declval<C>(), std::declval<K>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

