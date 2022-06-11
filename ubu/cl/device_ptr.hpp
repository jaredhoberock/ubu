#pragma once

#include "../detail/prologue.hpp"

#include "../memory/pointer/remote_ptr.hpp"
#include "../memory/plain_old_data.hpp"
#include "detail/throw_on_cl_error.hpp"
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <CL/cl.h>


namespace ubu::cl
{


template<plain_old_data_or_void T>
struct address
{
  cl_mem memory_object;
  std::size_t offset;

  auto operator<=>(const address&) const = default;
  
  using addressee_type = T;
  
  inline static address make_null_address()
  {
    return {};
  }
  
  template<class = void>
    requires (!std::is_void_v<T>)
  inline void advance_address(std::ptrdiff_t n)
  {
    offset += n;
  }
  
  template<class = void>
    requires (!std::is_void_v<T>)
  inline std::ptrdiff_t address_difference(const address& rhs) const
  {
    return offset - rhs.offset;
  }
};


class device_memory_copier
{
  public:
    template<plain_old_data_or_void T>
    using address = cl::address<T>;

    constexpr device_memory_copier(cl_command_queue queue)
      : queue_{queue}
    {}

    device_memory_copier(const device_memory_copier&) = default;

    constexpr device_memory_copier()
      : device_memory_copier{cl_command_queue{}}
    {}

    template<plain_old_data T>
    void copy_n(const T* from, std::size_t count, address<T> to) const
    {
      detail::throw_on_cl_error(
        clEnqueueWriteBuffer(queue_,
                             to.memory_object,      // destination buffer
                             CL_TRUE,               // blocking
                             sizeof(T) * to.offset, // destination offset
                             sizeof(T) * count,     // num bytes
                             from,                  // source address
                             0,                     // number of events to wait on
                             nullptr,               // pointer to array of events to wait on
                             nullptr),              // ignored return event
        "cl::device_memory_copier::copy_n: CL error after clEnqueueWriteBuffer"
      );
    }

    template<plain_old_data T>
    void copy_n(address<T> from, std::size_t count, std::remove_cvref_t<T>* to) const
    {
      detail::throw_on_cl_error(
        clEnqueueReadBuffer(queue_,
                            from.memory_object,      // source buffer
                            CL_TRUE,                 // blocking
                            sizeof(T) * from.offset, // source offset
                            sizeof(T) * count,       // num bytes
                            to,                      // destination address
                            0,                       // number of events to wait on
                            nullptr,                 // pointer to array of events to wait on
                            nullptr),                // ignored return event
        "cl::device_memory_copier::copy_n: CL error after clEnqueueReadBuffer"
      );
    }

    template<plain_old_data T>
    void copy_n(address<T> from, std::size_t count, address<std::remove_cvref_t<T>> to) const
    {
      cl_event copy_complete{};

      detail::throw_on_cl_error(
        clEnqueueCopyBuffer(queue_,
                            from.memory_object, // source buffer
                            to.memory_object,   // destination buffer
                            from.offset,        // source offset
                            to.offset,          // destination offset
                            sizeof(T) * count,  // num bytes
                            0,                  // number of events to wait on
                            nullptr,            // pointer to array of events to wait on
                            &copy_complete),    // return event
        "cl::device_memory_copier::copy_n: CL error after clEnqueueCopyBuffer"
      );

      // wait for copy to complete
      detail::throw_on_cl_error(
        clWaitForEvents(1, &copy_complete),
        "cl::device_memory_copier::copy_n: CL error after clWaitForEvents"
      );
    }

    constexpr bool operator==(const device_memory_copier& other) const
    {
      return queue_ == other.queue_;
    }

  private:
    cl_command_queue queue_;
};


template<plain_old_data_or_void T>
using device_ptr = remote_ptr<T, device_memory_copier>;


} // end ubu::cl


#include "../detail/epilogue.hpp"

