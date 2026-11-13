#include <string>

std::string string_conditional_value_min(bool choose_first,
                                         const std::string &first,
                                         const std::string &second) {
  return choose_first ? first : second;
}

