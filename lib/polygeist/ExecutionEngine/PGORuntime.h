
// TODO Remove the syncs and move PGO related stuff in a separate wrapper file.
// The syncs should instead be emitted by the code that emits the calls to the
// PGO functions which should know whether the code in the alternatives op is
// GPU code - we can add an attrib to the alternatives op for that

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <vector>

extern "C" int32_t mgpurtDeviceSynchronizeErr(void);

#ifdef _WIN32
#define MLIR_PGO_WRAPPERS_EXPORT __declspec(dllexport) __attribute__((weak))
#else
#define MLIR_PGO_WRAPPERS_EXPORT __attribute__((weak))
#endif // _WIN32

class PGOState {
public:
  enum Type { Start, End };
  struct State {
    struct timespec start_clock;
  };

  struct Logger {
    std::map<std::string, std::vector<double>> timings;
    ~Logger() { PGOState::writeResults(); }
  };

  inline static int alternative;
  inline static std::string dirname;
  inline thread_local static std::mutex mutex;
  inline thread_local static std::map<std::string, State *> states;
  inline static Logger logger;

  const char *kernelId_c;
  int totalAlternatives;

  std::string getKernelId() {
    std::string kernelId = kernelId_c;
    for (char &c : kernelId)
      if (c == '/')
        c = '+';
    return kernelId;
  }

  PGOState(const char *kernelId_c, int totalAlternatives)
      : totalAlternatives(totalAlternatives) {
    this->kernelId_c = kernelId_c;
  }
  void end() {
    struct timespec end_clock;
    mgpurtDeviceSynchronizeErr();
    clock_gettime(CLOCK_MONOTONIC, &end_clock);

    auto kernelId = getKernelId();
    std::unique_lock<std::mutex> lock(mutex);
    if (states.count(kernelId) == 0) {
      std::cerr << "No kernel with id " << kernelId << "running" << std::endl;
      exit(1);
    }
    State *state = states[kernelId];
    struct timespec tmp_clock {
      end_clock.tv_sec - state->start_clock.tv_sec,
          end_clock.tv_nsec - state->start_clock.tv_nsec
    };
    double elapsed =
        (tmp_clock.tv_sec + ((double)tmp_clock.tv_nsec) * .000000001);

    if (states.count(kernelId) == 0)
      logger.timings[kernelId] = {};
    logger.timings[kernelId].push_back(elapsed);

    delete state;
    states.erase(states.find(kernelId));
  }

  void start() {
    auto kernelId = getKernelId();
    std::unique_lock<std::mutex> lock(mutex);
    State *state = new State();
    if (states.count(kernelId) == 1) {
      std::cerr << "Two kernels with id " << kernelId
                << "running at the same time" << std::endl;
      exit(1);
    }
    states[kernelId] = state;
    // Start timing
    mgpurtDeviceSynchronizeErr();
    clock_gettime(CLOCK_MONOTONIC, &state->start_clock);
  }

  static void writeResults() {
    // Only write to file if we are profiling a valid alternative
    for (auto &pair : logger.timings) {
      auto &kernelId = std::get<0>(pair);
      auto &timings = std::get<1>(pair);
      auto elapsed = std::accumulate(timings.begin(), timings.end(), 0.0f);
      // TODO error handling
      std::ofstream ofile;
      ofile.open(std::string(dirname) + "/" + kernelId,
                 std::ios::out | std::ios::app);
      ofile << alternative << " " << elapsed << std::endl;
      ofile.close();
    }
  }

  int getAlternative() {
    static int init = [&] {
      if (char *i = getenv(POLYGEIST_PGO_ALTERNATIVE_ENV_VAR)) {
        this->alternative = atoi(i);
      } else {
        std::cerr << POLYGEIST_PGO_ALTERNATIVE_ENV_VAR << " not defined"
                  << std::endl;
        exit(1);
      }
      if (char *d = getenv(POLYGEIST_PGO_DATA_DIR_ENV_VAR)) {
        this->dirname = d;
      } else {
        this->dirname = POLYGEIST_PGO_DEFAULT_DATA_DIR;
      }
      std::filesystem::create_directories(dirname);

      return 0;
    }();
    return alternative % totalAlternatives;
  }

  ~PGOState() {}
};

extern "C" MLIR_PGO_WRAPPERS_EXPORT int32_t
mgpurtPGOGetAlternative(const char *kernelID, int totalAlternatives) {
  PGOState pgoState(kernelID, totalAlternatives);
  return pgoState.getAlternative();
}

extern "C" MLIR_PGO_WRAPPERS_EXPORT void mgpurtPGOStart(const char *kernelID,
                                                        int totalAlternatives) {
  PGOState pgoState(kernelID, totalAlternatives);
  pgoState.start();
}

extern "C" MLIR_PGO_WRAPPERS_EXPORT void mgpurtPGOEnd(const char *kernelID,
                                                      int totalAlternatives) {
  PGOState pgoState(kernelID, totalAlternatives);
  pgoState.end();
}
