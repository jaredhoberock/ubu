#pragma once

#include "../../detail/prologue.hpp"

#include "../../places/memory/pointers/remote_ptr.hpp"
#include "../../places/memory/plain_old_data.hpp"
#include "detail/throw_on_cl_error.hpp"
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <CL/cl.h>


namespace ubu::cl
{


struct address
{
  cl_mem memory_object;
  std::size_t offset;

  auto operator<=>(const address&) const = default;
  
  inline static address make_null_address()
  {
    return {};
  }
  
  inline void advance_address(std::ptrdiff_t num_bytes)
  {
    offset += num_bytes;
  }
  
  inline std::ptrdiff_t address_difference(const address& rhs) const
  {
    return offset - rhs.offset;
  }
};


class device_memory_loader
{
  public:
    using address_type = cl::address;

    constexpr device_memory_loader(cl_command_queue queue)
      : queue_{queue}
    {}

    device_memory_loader(const device_memory_loader&) = default;

    constexpr device_memory_loader()
      : device_memory_loader{cl_command_queue{}}
    {}

    void upload(const void* from, std::size_t num_bytes, address to) const
    {
      detail::throw_on_cl_error(
        clEnqueueWriteBuffer(queue_,
                             to.memory_object, // destination buffer
                             CL_TRUE,          // blocking
                             to.offset,        // destination offset
                             num_bytes,        // num bytes
                             from,             // source address
                             0,                // number of events to wait on
                             nullptr,          // pointer to array of events to wait on
                             nullptr),         // ignored return event
        "cl::device_memory_copier::copy_n: CL error after clEnqueueWriteBuffer"
      );
    }

    void download(address from, std::size_t num_bytes, void* to) const
    {
      detail::throw_on_cl_error(
        clEnqueueReadBuffer(queue_,
                            from.memory_object, // source buffer
                            CL_TRUE,            // blocking
                            from.offset,        // source offset
                            num_bytes,          // num bytes
                            to,                 // destination address
                            0,                  // number of events to wait on
                            nullptr,            // pointer to array of events to wait on
                            nullptr),           // ignored return event
        "cl::device_memory_copier::copy_n: CL error after clEnqueueReadBuffer"
      );
    }

    constexpr bool operator==(const device_memory_loader& other) const
    {
      return queue_ == other.queue_;
    }

  private:
    cl_command_queue queue_;
};


template<plain_old_data_or_void T>
using device_ptr = remote_ptr<T, device_memory_loader>;


} // end ubu::cl


#include "../../detail/epilogue.hpp"

