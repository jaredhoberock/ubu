#pragma once

#include "../../detail/prologue.hpp"

#include "detail/throw_on_cl_error.hpp"
#include <algorithm>
#include <concepts>
#include <CL/cl.h>


UBU_NAMESPACE_OPEN_BRACE


namespace cl
{


class event
{
  public:
    event() = default;

    inline event(cl_event native_handle)
      : native_handles_{1,native_handle}
    {}

    constexpr event(event&& other) noexcept
      : event{}
    {
      swap(other);
    }

    constexpr event& operator=(event&& other)
    {
      swap(other);
      return *this;
    }

    inline bool is_ready() const
    {
      return std::all_of(native_handles_.begin(), native_handles_.end(), [](cl_event e)
      {
        cl_int status{};
        detail::throw_on_cl_error(
          clGetEventInfo(e, CL_EVENT_COMMAND_EXECUTION_STATUS, &status, nullptr);
          "cl::event::is_complete: CL error after clGetEventInfo"
        );

        return status == CL_COMPLETE;
      });
    }

    inline void wait()
    {
      detail::throw_on_cl_error(
        clWaitForEvents(native_handles_.size(), native_handles_.data()),
        "cl::event::wait: CL error after clWaitForEvents"
      );
    }

    template<std::same_as<event>... Es>
    event make_contingent_event(const Es&... es) const
    {
      return {0, *this, es...};
    }

    constexpr static event make_complete_event()
    {
      return {};
    }

    constexpr void swap(event& other)
    {
      std::swap(native_handles_, other.native_handles_);
    }

  private:
    // this ctor is available to make_contingent_event
    // the int parameter distinguishes this ctor from a copy ctor
    template<std::same_as<event>... Es>
    event(int, const event& e, const Es&... es)
      : event{}
    {
      // copy the arguments' handles
      detail::for_each_arg([&](const event& e) mutable
      {
        native_handles_.insert(native_handles_.end(), e.begin(), e.end());
      }, e, es...);
    }

    std::vector<cl_event> native_handles_;
};


} // end cl


UBU_NAMESPACE_CLOSE_BRACE


#include "../../detail/epilogue.hpp"

