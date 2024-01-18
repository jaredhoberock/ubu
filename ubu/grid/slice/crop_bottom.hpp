#pragma once

#include "../../detail/prologue.hpp"
#include "../view.hpp"
#include <span>

namespace ubu
{
namespace detail
{

template<class G, class O>
concept has_crop_bottom_member_function = requires(G g, O o)
{
  g.crop_bottom(o);
};

template<class G, class O>
concept has_crop_bottom_free_function = requires(G g, O o)
{
  crop_bottom(g,o);
};

// this is the type of crop_bottom
struct dispatch_crop_bottom
{
  template<class G, class O>
    requires has_crop_bottom_member_function<G&&,O&&>
  constexpr auto operator()(G&& g, O&& new_origin) const
  {
    return std::forward<G>(g).crop_bottom(std::forward<O>(new_origin));
  }

  template<class G, class O>
    requires (not has_crop_bottom_member_function<G&&,O&&>
              and has_crop_bottom_free_function<G&&,O&&>)
  constexpr auto operator()(G&& g, O&& new_origin) const
  {
    return crop_bottom(std::forward<G>(g), std::forward<O>(new_origin));
  }

  template<class T>
  constexpr T* operator()(T* ptr, std::ptrdiff_t new_origin) const
  {
    return ptr + new_origin;
  }

  template<class T>
  constexpr std::span<T> operator()(const std::span<T>& s, std::size_t new_origin) const
  {
    return s.subspan(new_origin);
  }

  // XXX default case not yet implemented
  template<grid G>
    requires (not has_crop_bottom_member_function<G,grid_coordinate_t<G>>
              and not has_crop_bottom_free_function<G,grid_coordinate_t<G>>)
  constexpr G operator()(const G& g, const grid_coordinate_t<G>& new_origin) const = delete;
};

} // end detail


inline constexpr detail::dispatch_crop_bottom crop_bottom;


} // end ubu

#include "../../detail/epilogue.hpp"

