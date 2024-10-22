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


// the versions of load and store below because the version of LLVM used by circle,
// at low optimization levels, emits atomic operations which cannot
// be codegenned to device code in older LLVM


template<class T>
  requires (std::is_trivially_copy_constructible_v<T> and (sizeof(T) == sizeof(uint32_t)))
T load_acquire(volatile T* ptr)
{
  if UBU_TARGET(ubu::detail::is_device())
  {
    union {
      T result;
      uint32_t alias;
    } u{};
    asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(u.alias) : "l"((T*)ptr) : "memory");
    return u.result;
  }
  else
  {
    // XXX circle crashes on this alternative to the above:
    //     https://godbolt.org/z/qvMbh7G6G
    return std::atomic_ref(*ptr).load(std::memory_order_acquire);
  }
}


template<class T>
  requires (std::is_trivially_copy_constructible_v<T> and (sizeof(T) == sizeof(uint32_t)))
T load_relaxed(volatile T* ptr)
{
  if UBU_TARGET(ubu::detail::is_device())
  {
    union {
      T result;
      uint32_t alias;
    } u{};
    asm volatile("ld.relaxed.gpu.u32 %0,[%1];" : "=r"(u.alias) : "l"((T*)ptr) : "memory");
    return u.result;
  }
  else
  {
    // XXX circle crashes on this alternative to the above:
    //     https://godbolt.org/z/qvMbh7G6G
    return std::atomic_ref(*ptr).load(std::memory_order_relaxed);
  }
}


template<class T>
  requires (std::is_trivially_copy_constructible_v<T> and (sizeof(T) == sizeof(uint64_t)))
T load_relaxed(volatile T* ptr)
{
  if UBU_TARGET(ubu::detail::is_device())
  {
    union {
      T result;
      uint64_t alias;
    } u{};
    asm volatile("ld.relaxed.gpu.u64 %0,[%1];" : "=l"(u.alias) : "l"((T*)ptr) : "memory");
    return u.result;
  }
  else
  {
    // XXX circle crashes on this alternative to the above:
    //     https://godbolt.org/z/qvMbh7G6G
    return std::atomic_ref(*ptr).load(std::memory_order_relaxed);
  }
}



template<class T>
  requires (std::is_trivially_copyable_v<T> and (sizeof(T) == sizeof(uint32_t)))
void store_release(volatile T* ptr, T value)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (
    union {
      T value;
      uint32_t alias;
    } u{value};

    NV_IF_TARGET(NV_PROVIDES_SM_70, (
      asm volatile("st.release.gpu.u32 [%0],%1;" : : "l"((T*)ptr), "r"(u.alias) : "memory");
    ), (
      __threadfence();
      asm volatile("st.cg.u32 [%0],%1;" : : "l"((T*)ptr), "r"(u.alias) : "memory");
    ))
  ), (
    // XXX circle emits an LLVM ERROR : cannot select on this alternative to the above:
    std::atomic_ref(*ptr).store(value, std::memory_order_release);
  ))
}


template<class T>
  requires (std::is_trivially_copyable_v<T> and (sizeof(T) == sizeof(uint64_t)))
void store_release(volatile T* ptr, T value)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (
    union {
      T value;
      uint64_t alias;
    } u{value};

    NV_IF_TARGET(NV_PROVIDES_SM_70, (
      asm volatile("st.release.gpu.u64 [%0],%1;" : : "l"((T*)ptr), "r"(u.alias) : "memory");
    ), (
      __threadfence();
      asm volatile("st.cg.u32 [%0],%1;" : : "l"((T*)ptr), "r"(u.alias) : "memory");
    ))
  ), (
    // XXX circle emits an LLVM ERROR : cannot select on this alternative to the above:
    std::atomic_ref(*ptr).store(value, std::memory_order_release);
  ))
}


template<class T>
  requires (std::is_trivially_copyable_v<T> and (sizeof(T) == sizeof(uint32_t)))
inline void store_relaxed(volatile T* ptr, T value)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (
    NV_IF_TARGET(NV_PROVIDES_SM_70, (
      union {
        T value;
        uint32_t alias;
      } u{value};

      asm volatile("st.relaxed.gpu.u32 [%0],%1;" : : "l"((T*)ptr), "r"(u.alias) : "memory");
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


template<class T>
  requires (std::is_trivially_copyable_v<T> and (sizeof(T) == sizeof(uint64_t)))
inline void store_relaxed(volatile T* ptr, T value)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (
    NV_IF_TARGET(NV_PROVIDES_SM_70, (
      union {
        T value;
        uint64_t alias;
      } u{value};

      asm volatile("st.relaxed.gpu.u64 [%0],%1;" : : "l"((T*)ptr), "l"(u.alias) : "memory");
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

