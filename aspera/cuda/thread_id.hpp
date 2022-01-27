#pragma once

#include "../detail/prologue.hpp"


ASPERA_NAMESPACE_OPEN_BRACE

#include <vector_types.h>


namespace cuda
{


constexpr std::size_t size(const ::int3&)
{
  return 3;
}

template<std::size_t i>
constexpr const int& element(const ::int3&);

template<std::size_t i>
constexpr int& element(::int3&);

template<>
constexpr const int& element<0>(const ::int3& d)
{
  return d.x;
}

template<>
constexpr int& element<0>(::int3& d)
{
  return d.x;
}

template<>
constexpr const int& element<1>(const ::int3& d)
{
  return d.y;
}

template<>
constexpr int& element<1>(::int3& d)
{
  return d.y;
}

template<>
constexpr const int& element<2>(const ::int3& d)
{
  return d.z;
}

template<>
constexpr int& element<2>(::int3& d)
{
  return d.z;
}


struct thread_id
{
  ::int3 block;
  ::int3 thread;

  constexpr static std::size_t size()
  {
    return 2;
  }
};

template<std::size_t i>
constexpr const ::int3& element(const thread_id&);

template<std::size_t i>
constexpr ::int3& element(thread_id&);

template<>
constexpr const ::int3& element<0>(const thread_id& t)
{
  return t.block;
}

template<>
constexpr ::int3& element<0>(thread_id& t)
{
  return t.block;
}

template<>
constexpr const ::int3& element<1>(const thread_id& t)
{
  return t.thread;
}

template<>
constexpr ::int3& element<1>(thread_id& t)
{
  return t.thread;
}


} // end cuda


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

