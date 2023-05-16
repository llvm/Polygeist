#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "polygeist/Passes/Passes.h"

#if POLYGEIST_ENABLE_ROCM

#define DEBUG_TYPE "serialize-to-hsaco"

using namespace mlir;

namespace {
class SerializeToHsacoPass
    : public PassWrapper<SerializeToHsacoPass, gpu::SerializeToBlobPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerializeToHsacoPass)

  SerializeToHsacoPass(StringRef triple = "??? TODO",
                       bool outputIntermediate = false);

  StringRef getArgument() const override { return "gpu-to-hsaco"; }
  StringRef getDescription() const override {
    return "Lower GPU kernel function to HSACO binary annotations";
  }

protected:
  LogicalResult optimizeLlvm(llvm::Module &llvmModule,
                             llvm::TargetMachine &targetMachine) override;
  std::unique_ptr<llvm::Module>
  translateToLLVMIR(llvm::LLVMContext &llvmContext) override;

private:
  void getDependentDialects(DialectRegistry &registry) const override;

  // Serializes PTX to HSACO.
  std::unique_ptr<std::vector<char>>
  serializeISA(const std::string &isa) override;
};
} // namespace

SerializeToHsacoPass::SerializeToHsacoPass(StringRef triple,
                                           bool outputIntermediate) {}

LogicalResult
SerializeToHsacoPass::optimizeLlvm(llvm::Module &llvmModule,
                                   llvm::TargetMachine &targetMachine) {}
std::unique_ptr<llvm::Module>
SerializeToHsacoPass::translateToLLVMIR(llvm::LLVMContext &llvmContext) {}

void SerializeToHsacoPass::getDependentDialects(
    DialectRegistry &registry) const {}

// Serializes PTX to HSACO.
std::unique_ptr<std::vector<char>>
SerializeToHsacoPass::serializeISA(const std::string &isa) {}

namespace mlir::polygeist {

// Register pass to serialize GPU kernel functions to a HSACO binary annotation.
void registerGpuSerializeToHsacoPass() {
  PassRegistration<SerializeToHsacoPass> registerSerializeToHsaco([] {
    // TODO
    return std::make_unique<SerializeToHsacoPass>();
  });
}

std::unique_ptr<Pass> createGpuSerializeToHsacoPass(StringRef triple,
                                                    bool outputIntermediate) {
  return std::make_unique<SerializeToHsacoPass>(triple, outputIntermediate);
}

} // namespace mlir::polygeist

#else
namespace mlir::polygeist {
void registerGpuSerializeToHsacoPass() {}
} // namespace mlir::polygeist
#endif
