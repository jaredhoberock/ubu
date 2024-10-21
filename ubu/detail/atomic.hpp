#pragma once

#include "prologue.hpp"
#include "reflection/is_device.hpp"
#include <atomic>
#include <nv/target>
#include <type_traits>

// all of the functions in this file are a workaround for issues with circle/LLVM generating the correct
// memory operations for std::atomic_ref::load and ::store

namespace ubu::detail
{

// this is provided because circle ca. 04/2024 isn't generating atomic operations for user types
template<class T>
  requires (sizeof(T) == sizeof(std::uint64_t))
constexpr T atomic_load(const T* ptr, std::memory_order order = std::memory_order_seq_cst)
{
  union {
    T data;
    std::uint64_t alias;
  } result{};

  result.alias = std::atomic_ref(*reinterpret_cast<const std::uint64_t*>(ptr)).load(order);
  return result.data;
}


// this is provided because circle ca. 04/2024 isn't generating atomic operations for user types
template<class T>
  requires (sizeof(T) == sizeof(std::uint64_t))
constexpr void atomic_store(T* ptr, T desired, std::memory_order order = std::memory_order_seq_cst)
{
  union {
    T data;
    std::uint64_t alias;
  } store_me{};

  store_me.data = desired;

  std::atomic_ref(*reinterpret_cast<std::uint64_t*>(ptr)).store(store_me.alias, order);
}


// this is provided because circle ca. 04/2024 isn't generating atomic operations for user types
template<class T>
  requires (sizeof(T) == sizeof(std::uint32_t))
constexpr void atomic_store(T* ptr, T desired, std::memory_order order = std::memory_order_seq_cst)
{
  union {
    T data;
    std::uint32_t alias;
  } store_me{};

  store_me.data = desired;
  std::atomic_ref(*reinterpret_cast<std::uint32_t*>(ptr)).store(store_me.alias, order);
}


// this is provided because LLVM isn't generating ld.acquire
template<class T>
  requires (std::is_trivially_copy_constructible_v<T> and (sizeof(T) == sizeof(int)))
T load_acquire(volatile T* ptr)
{
  if UBU_TARGET(ubu::detail::is_device())
  {
    T result;
    asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(result) : "l"((T*)ptr) : "memory");
    return result;
  }
  else
  {
    // XXX circle crashes on this alternative to the above:
    //     https://godbolt.org/z/qvMbh7G6G
    return std::atomic_ref(*ptr).load(std::memory_order_acquire);
  }
}


// this is provided because LLVM isn't generating st.release
template<class T>
  requires (std::is_trivially_copyable_v<T> and (sizeof(T) == sizeof(int)))
void store_release(volatile T* ptr, T value)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (
    NV_IF_TARGET(NV_PROVIDES_SM_70, (
      asm volatile("st.release.gpu.u32 [%0],%1;" : : "l"((T*)ptr), "r"(value) : "memory");
    ), (
      __threadfence();
      asm volatile("st.cg.u32 [%0],%1;" : : "l"((T*)ptr), "r"(value) : "memory");
    ))
  ), (
    // XXX circle emits an LLVM ERROR : cannot select on this alternative to the above:
    std::atomic_ref(*ptr).store(value, std::memory_order_release);
  ))
}


// this is provided because at low optimization levels, the use of std::atomic_ref::store
// causes LLVM to attempt to emit store seq_cst and crashes with a can't select error
template<class T>
  requires (std::is_trivially_copyable_v<T> and (sizeof(T) == sizeof(uint32_t)))
inline void store_relaxed(volatile T* ptr, T value)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (
    NV_IF_TARGET(NV_PROVIDES_SM_70, (
      asm volatile("st.relaxed.gpu.u32 [%0],%1;" : : "l"((T*)ptr), "r"(value) : "memory");
    ), (
      // use a stronger store if we can't generate a relaxed store
      // XXX what should we actually do here for sm < 70?
      ubu::detail::store_release(ptr, value);
    ))
  ), (
    // XXX circle emits an LLVM ERROR : cannot select on this alternative to the above:
    std::atomic_ref(*ptr).store(value, std::memory_order_relaxed);
  ))
}


} // end ubu::detail

#include "epilogue.hpp"

