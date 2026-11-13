#include <mutex>

std::mutex global_mutex;

void std_mutex_global_lock_min() { global_mutex.lock(); }

void std_mutex_global_unlock_min() { global_mutex.unlock(); }

