#pragma once

#include "../detail/prologue.hpp"

#include "detail/throw_on_cl_error.hpp"
#include <vector>
#include <CL/cl.h>

UBU_NAMESPACE_OPEN_BRACE

namespace cl
{

class context
{
  public:
    // XXX in general, the context needs to know which devices to use
    inline context()
      : devices_{create_device_ids()},
        handle_{create_context(devices_)}
    {}

    inline context(cl_device_id device)
      : devices_{1, device},
        handle_{create_context(devices_)}
    {}

    inline context(context&& other)
      : context{}
    {
      std::swap(devices_, other.devices_);
      std::swap(handle_, other.handle_);
    }

    inline ~context()
    {
      clReleaseContext(handle_);
    }

    constexpr cl_context native_handle() const
    {
      return handle_;
    }

    cl_device_id device(int i) const
    {
      return devices_[i];
    }

  private:
    inline std::vector<cl_device_id> create_device_ids()
    {
      // choose the first available platform
      cl_platform_id platform = 0;
      detail::throw_on_cl_error(
        clGetPlatformIDs(1, &platform, nullptr),
        "cl::context::create_context: CL error after clGetPlatformIDs"
      );

      // first count the number of devices
      cl_uint num_devices = 0;
      detail::throw_on_cl_error(
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 0, nullptr, &num_devices),
        "cl::context::create_context: CL error after clGetDeviceIDs"
      );

      if(num_devices == 0)
      {
        throw std::runtime_error("cl::context::create_context: No available devices");
      }

      // get the devices
      std::vector<cl_device_id> result{num_devices};
      detail::throw_on_cl_error(
        clGetDeviceIDs(platform,       
                       CL_DEVICE_TYPE_DEFAULT,
                       num_devices,
                       result.data(),
                       nullptr),
        "cl::context::create_context: CL error after clGetDeviceIDs"
      );

      return result;
    }

    inline static cl_context create_context(const std::vector<cl_device_id>& devices)
    {
      // create the context
      cl_int error = 0;
      cl_context result = clCreateContext(nullptr,        // properties (none)
                                          devices.size(), // num_devices
                                          devices.data(), // devices
                                          nullptr,        // pfn_notify callback
                                          nullptr,        // user_data for notify callback
                                          &error);        // errcode_ret

      if(error)
      {
        clReleaseContext(result);
        detail::throw_on_cl_error(error, "cl::context::create_context: CL error after clCreateContext");
      }

      return result;
    }

    std::vector<cl_device_id> devices_;
    cl_context handle_;
};

} // end cl

UBU_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

