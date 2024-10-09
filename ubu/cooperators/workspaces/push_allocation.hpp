#pragma once

#include "../../detail/prologue.hpp"
#include "../../places/memory/data.hpp"
#include "../../places/memory/views/get_buffer.hpp"
#include "detail/advance_data.hpp"

namespace ubu
{
namespace detail
{


template<class W, class N>
concept has_push_allocation_member_function = requires(W w, N n)
{
  { std::forward<W>(w).push_allocation(std::forward<N>(n)) } -> pointer_like;
};

template<class W, class N>
concept has_push_allocation_free_function = requires(W w, N n)
{
  { push_allocation(std::forward<W>(w), std::forward<N>(n)) } -> pointer_like;
};

template<class W, class N>
concept has_push_allocation_customization = has_push_allocation_member_function<W,N> or has_push_allocation_free_function<W,N>;


struct dispatch_push_allocation
{
  template<class W, class N>
    requires has_push_allocation_customization<W&&,N&&>
  constexpr pointer_like auto operator()(W&& ws, N&& n) const
  {
    if constexpr (has_push_allocation_member_function<W&&,N&&>)
    {
      return std::forward<W>(ws).push_allocation(std::forward<N>(n));
    }
    else
    {
      return push_allocation(std::forward<W>(ws), std::forward<N>(n));
    }
  }

  // XXX ideally, we would make this work for any memory_view
  template<workspace W>
    requires (not has_push_allocation_member_function<W&,int>
              and detail::advanceable_span_like<buffer_t<W>,int>)
  constexpr pointer_like auto operator()(W& ws, int n) const
  {
    buffer_like auto& buffer = get_buffer(ws);
    pointer_like auto result = data(buffer);
    detail::advance_data(buffer, n);
    return result;
  }
};


} // end detail

constexpr inline detail::dispatch_push_allocation push_allocation;

} // end ubu

#include "../../detail/epilogue.hpp"

