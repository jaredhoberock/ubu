#pragma once

#include "../detail/prologue.hpp"

#include "../memory/fancy_ptr.hpp"
#include "detail/throw_on_cl_error.hpp"
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <CL/cl.h>


namespace ubu::cl
{


template<class T>
concept plain_old_data = std::is_standard_layout_v<T> and std::is_trivial_v<T>;


template<plain_old_data T>
class device_memory_copier
{
  public:
    using handle_type = std::pair<cl_mem, std::size_t>;

    using element_type = T;
    using value_type = std::remove_cv_t<T>;

    constexpr device_memory_copier(cl_command_queue queue)
      : queue_{queue}
    {}

    device_memory_copier(const device_memory_copier&) = default;

    constexpr device_memory_copier()
      : device_memory_copier{cl_command_queue{}}
    {}

    static handle_type null_handle()
    {
      return {};
    }

    static void advance(handle_type& h, std::ptrdiff_t n)
    {
      h.second += n;
    }

    handle_type copy_n_from_raw_pointer(const T* from, std::size_t count, handle_type to) const
    {
      detail::throw_on_cl_error(
        clEnqueueWriteBuffer(queue_,
                             to.first,              // destination buffer
                             CL_TRUE,               // blocking
                             sizeof(T) * to.second, // destination offset
                             sizeof(T) * count,     // num bytes
                             from,                  // source address
                             0,                     // number of events to wait on
                             nullptr,               // pointer to array of events to wait on
                             nullptr),              // ignored return event
        "cl::device_memory_copier::copy_n_from_raw_pointer: CL error after clEnqueueWriteBuffer"
      );

      return {to.first, to.second + count};
    }

    std::remove_cv_t<T>* copy_n_to_raw_pointer(handle_type from, std::size_t count, std::remove_cv_t<T>* to) const
    {
      detail::throw_on_cl_error(
        clEnqueueReadBuffer(queue_,
                            from.first,              // source buffer
                            CL_TRUE,                 // blocking
                            sizeof(T) * from.second, // source offset
                            sizeof(T) * count,       // num bytes
                            to,                      // destination address
                            0,                       // number of events to wait on
                            nullptr,                 // pointer to array of events to wait on
                            nullptr),                // ignored return event
        "cl::device_memory_copier::copy_n_to_raw_pointer: CL error after clEnqueueReadBuffer"
      );

      return to + count;
    }

    handle_type copy_n(handle_type from, std::size_t count, handle_type to) const
    {
      cl_event copy_complete{};

      detail::throw_on_cl_error(
        clEnqueueCopyBuffer(queue_,
                            from.first,        // source buffer
                            to.first,          // destination buffer
                            from.second,       // source offset
                            to.second,         // destination offset
                            sizeof(T) * count, // num bytes
                            0,                 // number of events to wait on
                            nullptr,           // pointer to array of events to wait on
                            &copy_complete),   // return event
        "cl::device_memory_copier::copy_n: CL error after clEnqueueCopyBuffer"
      );

      // wait for copy to complete
      detail::throw_on_cl_error(
        clWaitForEvents(1, &copy_complete),
        "cl::device_memory_copier::copy_n: CL error after clWaitForEvents"
      );

      return {to.first, to.second + count};
    }

  private:
    cl_command_queue queue_;
};


template<plain_old_data T>
using device_ptr = fancy_ptr<T, device_memory_copier<T>>;


} // end ubu::cl


#include "../detail/epilogue.hpp"

