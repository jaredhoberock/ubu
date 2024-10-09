#pragma once

#include "../../detail/prologue.hpp"
#include "../../places/memory/views/get_buffer.hpp"
#include "detail/advance_data.hpp"

namespace ubu
{
namespace detail
{


template<class W, class N>
concept has_pop_allocation_member_function = requires(W w, N n)
{
  std::forward<W>(w).pop_allocation(std::forward<N>(n));
};

template<class W, class N>
concept has_pop_allocation_free_function = requires(W w, N n)
{
  pop_allocation(std::forward<W>(w), std::forward<N>(n));
};

template<class W, class N>
concept has_pop_allocation_customization = has_pop_allocation_member_function<W,N> or has_pop_allocation_free_function<W,N>;


struct dispatch_pop_allocation
{
  template<class W, class N>
    requires has_pop_allocation_customization<W&&,N&&>
  constexpr void operator()(W&& ws, N&& n) const
  {
    if constexpr (has_pop_allocation_member_function<W&&,N&&>)
    {
      std::forward<W>(ws).pop_allocation(std::forward<N>(n));
    }
    else
    {
      pop_allocation(std::forward<W>(ws), std::forward<N>(n));
    }
  }

  // XXX ideally, we would make this work for any memory_view
  template<workspace W>
    requires (not has_pop_allocation_member_function<W&,int>
              and detail::advanceable_span_like<buffer_t<W>,int>)
  constexpr void operator()(W& ws, int n) const
  {
    detail::advance_data(get_buffer(ws), -n);
  }
};


} // end detail

constexpr inline detail::dispatch_pop_allocation pop_allocation;

} // end ubu

#include "../../detail/epilogue.hpp"

