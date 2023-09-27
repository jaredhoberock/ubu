#pragma once

#include "../../detail/prologue.hpp"

#include "../../grid/coordinate/coordinate.hpp"
#include "../../grid/coordinate/point.hpp"
#include <iostream>


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

  constexpr thread_id(int2 thread_and_block)
    : thread{thread_and_block.x, 0, 0},
      block{thread_and_block.y, 0, 0}
  {}

  constexpr static std::size_t rank()
  {
    return 2;
  }

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

template<std::size_t i>
constexpr const int3& element(const thread_id&);

template<std::size_t i>
constexpr int3& element(thread_id&);

template<>
constexpr const int3& element<0>(const thread_id& id)
{
  return id.thread;
}

template<>
constexpr int3& element<0>(thread_id& id)
{
  return id.thread;
}

template<>
constexpr const int3& element<1>(const thread_id& id)
{
  return id.block;
}

template<>
constexpr int3& element<1>(thread_id& id)
{
  return id.block;
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

