#pragma once

#include "../../detail/prologue.hpp"

#include <utility>

namespace ubu
{
namespace detail
{


template<class T>
concept has_get_local_barrier_member_function = requires(T arg)
{
  { arg.get_local_barrier() } -> barrier_like;
};

template<class T>
concept has_get_local_barrier_free_function = requires(T arg)
{
  { get_local_barrier(arg) } -> barrier_like;
};

struct dispatch_get_local_barrier
{
  template<class T>
    requires has_get_local_barrier_member_function<T&&>
  constexpr decltype(auto) operator()(T&& arg) const
  {
    return std::forward<T>(arg).get_local_barrier();
  }

  template<class T>
    requires (not has_get_local_barrier_member_function<T&&>
              and has_get_local_barrier_free_function<T&&>)
  constexpr decltype(auto) operator()(T&& arg) const
  {
    return get_local_barrier(std::forward<T>(arg));
  }
};


} // end detail


namespace
{

// XXX this should maybe be something like arrive_and_wait_and_get_local_barrier
//     or something more like MPI_comm_split
// XXX anyway, the name of this isn't great
constexpr detail::dispatch_get_local_barrier get_local_barrier;

} // end anonymous namespace


template<class T>
using local_barrier_t = std::remove_cvref_t<decltype(get_local_barrier(std::declval<T>()))>;


} // end ubu

#include "../../detail/epilogue.hpp"


