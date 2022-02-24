#pragma once

#include "../detail/prologue.hpp"

#include "../coordinate/grid_coordinate.hpp"
#include "../coordinate/point.hpp"
#include <vector_types.h>


ASPERA_NAMESPACE_OPEN_BRACE


namespace cuda
{


struct thread_id
{
  int3 block;
  int3 thread;

  thread_id() = default;

  thread_id(const thread_id&) = default;

  thread_id(int3 b, int3 t)
    : block{b}, thread{t}
  {}

  thread_id(int2 block_and_thread)
    : block{block_and_thread.x, 0, 0},
      thread{block_and_thread.y, 0, 0}
  {}

  constexpr static std::size_t size()
  {
    return 2;
  }
};

template<std::size_t i>
constexpr const int3& element(const thread_id&);

template<std::size_t i>
constexpr int3& element(thread_id&);

template<>
constexpr const int3& element<0>(const thread_id& t)
{
  return t.block;
}

template<>
constexpr int3& element<0>(thread_id& t)
{
  return t.block;
}

template<>
constexpr const int3& element<1>(const thread_id& t)
{
  return t.thread;
}

template<>
constexpr int3& element<1>(thread_id& t)
{
  return t.thread;
}


} // end cuda


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

