#ifndef CG_OPTIONS_H
#define CG_OPITIOS_H

#include <string>
using namespace std;

struct CGOptions {

  bool FOpenMP = false;
  bool ShowAst = false;
  bool CudaLower = false;
  bool Verbose = false;

  string CUDAGPUArch = "";
  string CUDAPath = "";
  string MArch = "";
  string ResourceDir = "";
  string Standard = "";
};
#endif
