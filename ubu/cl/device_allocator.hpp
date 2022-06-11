#pragma once

#include "../detail/prologue.hpp"

#include "detail/throw_on_cl_error.hpp"
#include "device_ptr.hpp"
#include <cstddef>
#include <utility>
#include <stdexcept>
#include <CL/cl.h>


namespace ubu::cl
{


template<class T>
class device_allocator
{
  public:
    using value_type = T;
    using pointer = device_ptr<T>;

    explicit device_allocator(cl_context ctx, cl_command_queue queue)
      : context_{ctx},
        queue_{queue}
    {}

    device_allocator(const device_allocator&) = default;

    template<class OtherT>
    device_allocator(const device_allocator& other)
      : device_allocator{other.device()}
    {}

    device_ptr<T> allocate(std::size_t n) const
    {
      cl_int error = CL_SUCCESS;
      cl_mem buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(T) * n, nullptr, &error);
      detail::throw_on_cl_error(error, "cl::device_allocator::allocate: CL error after clCreateBuffer");

      // indicate the buffer's affinity
      cl_event migration_complete{};
      detail::throw_on_cl_error(
        clEnqueueMigrateMemObjects(queue_,
                                   1, &buffer, // one buffer to migrate
                                   CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                                   0, nullptr, // no events to wait on
                                   &migration_complete),
        "cl::device_allocator::allocate: CL error after clEnqueueMigrateMemObjects"
      );

      // wait for the memory to migrate
      detail::throw_on_cl_error(
        clWaitForEvents(1, &migration_complete),
        "cl::device_allocator::allocate: CL error after clWaitForEvents"
      );

      address<T> addr{buffer,0};

      return {addr, queue_};
    }

    void deallocate(pointer ptr, std::size_t) const
    {
      address<T> addr = ptr.to_address();

      if(addr.offset != 0)
      {
        throw std::runtime_error("cl::device_allocator::deallocate: Invalid pointer");
      }

      detail::throw_on_cl_error(clReleaseMemObject(addr.memory_object), "cl::device_allocator::deallocate: CL error after clReleaseBuffer");
    }

    cl_context context() const
    {
      return context_;
    }

    cl_command_queue command_queue() const
    {
      return queue_;
    }

    bool operator==(const device_allocator& other) const
    {
      return context() == other.context() and command_queue() == other.command_queue();
    }

    bool operator!=(const device_allocator& other) const
    {
      return !(*this == other);
    }

  private:
    cl_context context_;
    cl_command_queue queue_;
};


} // end ubu::cl


#include "../detail/epilogue.hpp"

