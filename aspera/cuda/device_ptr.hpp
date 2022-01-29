#pragma once

#include "../detail/prologue.hpp"

#include "../detail/exception.hpp"
#include "../memory/pointer_adaptor.hpp"
#include "detail/temporarily_with_current_device.hpp"
#include <type_traits>


ASPERA_NAMESPACE_OPEN_BRACE


namespace cuda
{


template<class T>
  requires std::is_pod_v<T>
class device_memory_copier
{
  public:
    using handle_type = T*;
    using element_type = T;
    using value_type = std::remove_cv_t<T>;

    constexpr device_memory_copier(int device)
      : device_{device}
    {}

    constexpr device_memory_copier()
      : device_memory_copier{0}
    {}

    device_memory_copier(const device_memory_copier&) = default;

    value_type* copy_n(const value_type* from, std::size_t count, value_type* to) const
    {
      detail::temporarily_with_current_device(device_, [=]
      {
        detail::throw_on_error(cudaMemcpy(to, from, sizeof(T), cudaMemcpyDefault),
          "device_memory_copier: CUDA error after cudaMemcpy"
        );
      });

      return to + count;
    }

    value_type* copy_n_to_raw_pointer(const value_type* from, std::size_t count, value_type* to) const
    {
      return this->copy_n(from, count, to);
    }

    value_type* copy_n_from_raw_pointer(const value_type* from, std::size_t count, value_type* to) const
    {
      return this->copy_n(from, count, to);
    }

    constexpr int device() const
    {
      return device_;
    }

  private:
    int device_;
};


template<class T>
using device_ptr = pointer_adaptor<T, device_memory_copier<T>>;


// copy_n overloads

template<class T>
  requires std::is_assignable_v<T,T>
device_ptr<T> copy_n(const std::remove_cv_t<T>* first, std::size_t count, device_ptr<T> result)
{
  return result.copier().copy_n_from_raw_pointer(first, count, result.native_handle());
}

template<class T>
  requires std::is_assignable_v<T,T>
std::remove_cv_t<T>* copy_n(device_ptr<T> first, std::size_t count, std::remove_cv_t<T>* result)
{
  return first.copier().copy_n_to_raw_pointer(first.native_handle(), count, result);
}

template<class T>
  requires std::is_assignable_v<T,T>
device_ptr<std::remove_cv_t<T>> copy_n(device_ptr<T> first, std::size_t count, device_ptr<std::remove_cv_t<T>> result)
{
  return first.copier().copy_n(first.native_handle(), count, result.native_handle());
}


} // end cuda

ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

