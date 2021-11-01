This directory contains functionality related to abnormal program termination:

* `terminate.hpp` - Defines the `detail::terminate` function.
* `terminate_with_message.hpp` - Defines the `detail::terminate_with_message` function.
* `throw_runtime_error.hpp` - Defines the `detail::throw_runtime_error` function.

This functionality provides abnormal program termination during GPU execution and exposes it in a target-agnostic interface.

