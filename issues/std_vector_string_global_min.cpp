#include <string>
#include <vector>

static std::vector<std::string> styles = {"low", "mid", "high"};

const char *std_vector_string_global_min(int index) {
  return styles[index].c_str();
}

