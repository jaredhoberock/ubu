#pragma once

#include "../../detail/prologue.hpp"
#include "../../miscellaneous/integral/integral_like.hpp"
#include "../uninitialized_coop_array.hpp"
#include "concepts/allocating_cooperator.hpp"
#include "synchronize.hpp"
#include "synchronize_and_any.hpp"
#include <optional>

namespace ubu
{


template<allocating_cooperator S, integral_like I, class T>
constexpr T broadcast(S self, const I& broadcaster, const T& message)
{
  uninitialized_coop_array<T,S> smem(self, 1_c);

  if(id(self) == broadcaster)
  {
    smem.construct_at(0, message);
  }

  synchronize(self);

  T result = smem[0];

  synchronize(self);

  return result;
}

template<allocating_cooperator S, integral_like I, class T>
constexpr std::optional<T> broadcast(S self, const I& broadcaster, const std::optional<T>& message)
{
  uninitialized_coop_array<T,S> smem(self, 1_c);

  if(id(self) == broadcaster)
  {
    if(message)
    {
      smem[0] = *message;
    }
  }

  std::optional<T> result;

  // did the broadcaster actually send a non-null message?
  if(synchronize_and_any(self, id(self) == broadcaster ? message.has_value() : false))
  {
    result = smem[0];
  }

  synchronize(self);

  return result;
}


} // end ubu

#include "../../detail/epilogue.hpp"

