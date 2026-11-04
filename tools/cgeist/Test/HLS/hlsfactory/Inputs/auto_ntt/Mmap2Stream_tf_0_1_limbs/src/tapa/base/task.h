#ifndef TAPA_BASE_TASK_H_
#define TAPA_BASE_TASK_H_

namespace tapa {

inline constexpr int join = 0;
inline constexpr int detach = -1;

/// Class that generates a sequence of integers as task arguments.
///
/// Canonical usage:
/// @code{.cpp}
///  void TaskFoo(int i, ...) {
///    ...
///  }
///  tapa::task()
///    .invoke<3>(TaskFoo, tapa::seq(), ...)
///    ...
///    ;
/// @endcode
///
/// @c TaskFoo will be invoked three times, receiving @c 0, @c 1, and @c 2 as
/// the first argument, respectively.
struct seq {
  /// Constructs a @c tapa::seq. This is the only public API.
  seq() = default;

  seq(const seq&) = delete;
  seq(seq&&) = delete;
  seq& operator=(const seq&) = delete;
  seq& operator=(seq&&) = delete;
  int pos = 0;
};

}  // namespace tapa

#endif  // TAPA_BASE_TASK_H_
