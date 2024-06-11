#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/reflection/is_device.hpp"
#include "../../detail/exception/throw_runtime_error.hpp"
#include "../../tensor/coordinates/concepts/coordinate.hpp"
#include "../../tensor/coordinates/coordinate_cast.hpp"
#include "../../tensor/coordinates/point.hpp"
#include <iostream>
#include <utility>


namespace ubu::cuda
{


struct thread_id
{
  int3 thread;
  int3 block;

  thread_id() = default;

  thread_id(const thread_id&) = default;

  constexpr thread_id(int3 t, int3 b)
    : thread{t}, block{b}
  {}

  template<congruent<int3> T, congruent<int3> B>
  constexpr thread_id(const T& thread, const B& block)
    : thread_id{coordinate_cast<int3>(thread), coordinate_cast<int3>(block)}
  {}

  // XXX eliminate this
  constexpr thread_id(int2 thread_and_block)
    : thread{thread_and_block.x, 0, 0},
      block{thread_and_block.y, 0, 0}
  {}

  // use pair<int3,int3> instead of block_id because thread_id is not yet complete
  template<congruent<std::pair<int3,int3>> C>
  constexpr thread_id(const C& thread_and_block)
    : thread_id{get<0>(thread_and_block), get<1>(thread_and_block)}
  {}

  auto operator<=>(const thread_id&) const = default;

  // tuple-like interface
  template<std::size_t i>
    requires (i < 2)
  friend constexpr int3& get(thread_id& self)
  {
    if constexpr (i == 0) return self.thread;
    return self.block;
  }

  template<std::size_t i>
    requires (i < 2)
  friend constexpr const int3& get(const thread_id& self)
  {
    if constexpr (i == 0) return self.thread;
    return self.block;
  }

  template<std::size_t i>
    requires (i < 2)
  friend constexpr int3&& get(thread_id&& self)
  {
    if constexpr (i == 0) return std::move(self.thread);
    return std::move(self.block);
  }
};


inline thread_id this_thread_id()
{
#if defined(__CUDACC__)
  if UBU_TARGET(ubu::detail::is_device())
  {
    return {{threadIdx.x, threadIdx.y, threadIdx.z}, {blockIdx.x, blockIdx.y, blockIdx.z}};
  }
  else
  {
    ubu::detail::throw_runtime_error("cuda::this_thread_id is unavailable in host code");
    return {};
  }
#else
  ubu::detail::throw_runtime_error("cuda::this_thread_id requires CUDA C++ language support.");
  return {};
#endif
}


std::ostream& operator<<(std::ostream& os, const thread_id& id)
{
  return os << "(" << id.thread << ", " << id.block << ")";
}


} // end ubu::cuda


namespace std
{

// additional tuple-like interface

template<>
struct tuple_size<ubu::cuda::thread_id> : std::integral_constant<size_t,2> {};

template<std::size_t I>
struct tuple_element<I,ubu::cuda::thread_id>
{
  using type = ubu::int3;
};

} // end std


#include "../../detail/epilogue.hpp"

