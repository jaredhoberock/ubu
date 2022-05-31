#pragma once

#include "../detail/prologue.hpp"

#include "detail/throw_on_cl_error.hpp"
#include <utility>
#include <stdexcept>
#include <CL/cl.h>

UBU_NAMESPACE_OPEN_BRACE


namespace cl
{


class device_memory_resource
{
  public:
    explicit device_memory_resource(cl_context ctx, cl_command_queue queue)
      : context_{ctx},
        queue_{queue}
    {}

    device_memory_resource(const device_memory_resource&) = default;

    inline std::pair<cl_mem,std::size_t> allocate(std::size_t num_bytes) const
    {
      cl_int error = CL_SUCCESS;
      cl_mem buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, num_bytes, nullptr, &error);
      detail::throw_on_cl_error(error, "cl::device_memory_resource::allocate: CL error after clCreateBuffer");

      // indicate the buffer's affinity
      cl_event migration_complete{};
      detail::throw_on_cl_error(
        clEnqueueMigrateMemObjects(queue_,
                                   1, &buffer, // one buffer to migrate
                                   CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                                   0, nullptr, // no events to wait on
                                   &migration_complete),
        "cl::device_memory_resource::allocate: CL error after clEnqueueMigrateMemObjects"
      );

      // wait for the memory to migrate
      detail::throw_on_cl_error(
        clWaitForEvents(1, &migration_complete),
        "cl::device_memory_resource::allocate: CL error after clWaitForEvents"
      );

      return {buffer,0};
    }

    inline void deallocate(std::pair<cl_mem,std::size_t> ptr, std::size_t) const
    {
      if(ptr.second != 0)
      {
        throw std::runtime_error("cl::device_memory_resource::deallocate: Invalid pointer");
      }

      detail::throw_on_cl_error(clReleaseMemObject(ptr.first), "cl::device_memory_resource::deallocate: CL error after clReleaseBuffer");
    }

    inline cl_context context() const
    {
      return context_;
    }

    inline cl_command_queue command_queue() const
    {
      return queue_;
    }

    inline bool is_equal(const device_memory_resource& other) const
    {
      return context() == other.context() and command_queue() == other.command_queue();
    }

    inline bool operator==(const device_memory_resource& other) const
    {
      return is_equal(other);
    }

    inline bool operator!=(const device_memory_resource& other) const
    {
      return !(*this == other);
    }

  private:
    cl_context context_;
    cl_command_queue queue_;
};


} // end cl


UBU_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

