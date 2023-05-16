#ifndef POLYGEIST_PASSES_RUNTIMEWRAPPERUTILS_H_
#define POLYGEIST_PASSES_RUNTIMEWRAPPERUTILS_H_

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

namespace {
struct FunctionCallBuilder {
  FunctionCallBuilder(StringRef functionName, Type returnType,
                      ArrayRef<Type> argumentTypes)
      : functionName(functionName),
        functionType(LLVM::LLVMFunctionType::get(returnType, argumentTypes)) {}
  LLVM::CallOp create(Location loc, OpBuilder &builder,
                      ArrayRef<Value> arguments) const;

  StringRef functionName;
  LLVM::LLVMFunctionType functionType;
};

LLVM::CallOp FunctionCallBuilder::create(Location loc, OpBuilder &builder,
                                         ArrayRef<Value> arguments) const {
  auto module = builder.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto function = [&] {
    if (auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(functionName))
      return function;
    return OpBuilder::atBlockEnd(module.getBody())
        .create<LLVM::LLVMFuncOp>(loc, functionName, functionType);
  }();
  return builder.create<LLVM::CallOp>(loc, function, arguments);
}

class GpuRuntimeCallBuildersContextHolder {
protected:
  GpuRuntimeCallBuildersContextHolder(MLIRContext *context,
                                      LLVMTypeConverter &typeConverter)
      : holder_context(context), holder_typeConverter(typeConverter){};
  MLIRContext *holder_context;
  LLVMTypeConverter &holder_typeConverter;
};
class GpuRuntimeCallBuilders : private GpuRuntimeCallBuildersContextHolder {

protected:
  GpuRuntimeCallBuilders(MLIRContext *context, LLVMTypeConverter &typeConverter)
      : GpuRuntimeCallBuildersContextHolder(context, typeConverter) {}

  MLIRContext *context = holder_context;

  Type llvmVoidType = LLVM::LLVMVoidType::get(context);
  Type llvmPointerType =
      LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  Type llvmPointerPointerType = LLVM::LLVMPointerType::get(llvmPointerType);
  Type llvmInt8Type = IntegerType::get(context, 8);
  Type llvmInt32Type = IntegerType::get(context, 32);
  Type llvmInt64Type = IntegerType::get(context, 64);
  Type llvmIntPtrType =
      IntegerType::get(context, holder_typeConverter.getPointerBitwidth(0));

  FunctionCallBuilder rtRegisterFunctionCallBuilder = {
      "__cudaRegisterFunction",
      llvmInt32Type,
      {llvmPointerPointerType, llvmPointerType, llvmPointerType,
       llvmPointerType, llvmInt32Type, llvmPointerType, llvmPointerType,
       llvmPointerType, llvmPointerType,
       llvmPointerType /* should actually be a pointer to int */}};
  FunctionCallBuilder rtUnregisterFatBinaryCallBuilder = {
      "__cudaUnregisterFatBinary", llvmVoidType, {llvmPointerPointerType}};
  FunctionCallBuilder rtRegisterFatBinaryCallBuilder = {
      "__cudaRegisterFatBinary", llvmPointerPointerType, {llvmPointerType}};
  FunctionCallBuilder rtRegisterFatBinaryEndCallBuilder = {
      "__cudaRegisterFatBinaryEnd", llvmVoidType, {llvmPointerPointerType}};
  FunctionCallBuilder rtAllocCallBuilder = {
      "mgpurtMemAlloc",
      llvmPointerType /* void * */,
      {llvmIntPtrType /* intptr_t sizeBytes */,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder allocCallBuilder = {
      "mgpuMemAlloc",
      llvmPointerType /* void * */,
      {llvmIntPtrType /* intptr_t sizeBytes */,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder moduleLoadCallBuilder = {
      "mgpuModuleLoad",
      llvmPointerType /* void *module */,
      {llvmPointerType /* void *cubin */}};
  FunctionCallBuilder moduleUnloadCallBuilder = {
      "mgpuModuleUnload", llvmVoidType, {llvmPointerType /* void *module */}};
  FunctionCallBuilder moduleGetFunctionCallBuilder = {
      "mgpuModuleGetFunction",
      llvmPointerType /* void *function */,
      {
          llvmPointerType, /* void *module */
          llvmPointerType  /* char *name   */
      }};
  FunctionCallBuilder launchKernelCallBuilder = {
      "mgpuLaunchKernel",
      llvmVoidType,
      {
          llvmPointerType,        /* void* f */
          llvmIntPtrType,         /* intptr_t gridXDim */
          llvmIntPtrType,         /* intptr_t gridyDim */
          llvmIntPtrType,         /* intptr_t gridZDim */
          llvmIntPtrType,         /* intptr_t blockXDim */
          llvmIntPtrType,         /* intptr_t blockYDim */
          llvmIntPtrType,         /* intptr_t blockZDim */
          llvmInt32Type,          /* unsigned int sharedMemBytes */
          llvmPointerType,        /* void *hstream */
          llvmPointerPointerType, /* void **kernelParams */
          llvmPointerPointerType  /* void **extra */
      }};
  FunctionCallBuilder launchKernelErrCallBuilder = {
      "mgpuLaunchKernelErr",
      llvmInt32Type, /* unsigned int */
      {
          llvmPointerType,        /* void* f */
          llvmIntPtrType,         /* intptr_t gridXDim */
          llvmIntPtrType,         /* intptr_t gridyDim */
          llvmIntPtrType,         /* intptr_t gridZDim */
          llvmIntPtrType,         /* intptr_t blockXDim */
          llvmIntPtrType,         /* intptr_t blockYDim */
          llvmIntPtrType,         /* intptr_t blockZDim */
          llvmInt32Type,          /* unsigned int sharedMemBytes */
          llvmPointerType,        /* void *hstream */
          llvmPointerPointerType, /* void **kernelParams */
          llvmPointerPointerType  /* void **extra */
      }};
  FunctionCallBuilder rtLaunchKernelCallBuilder = {
      "mgpurtLaunchKernel",
      llvmVoidType,
      {
          llvmPointerType,        /* void* f */
          llvmIntPtrType,         /* intptr_t gridXDim */
          llvmIntPtrType,         /* intptr_t gridyDim */
          llvmIntPtrType,         /* intptr_t gridZDim */
          llvmIntPtrType,         /* intptr_t blockXDim */
          llvmIntPtrType,         /* intptr_t blockYDim */
          llvmIntPtrType,         /* intptr_t blockZDim */
          llvmInt32Type,          /* unsigned int sharedMemBytes */
          llvmPointerType,        /* void *hstream */
          llvmPointerPointerType, /* void **kernelParams */
      }};
  FunctionCallBuilder rtLaunchKernelErrCallBuilder = {
      "mgpurtLaunchKernelErr",
      llvmInt32Type,
      {
          llvmPointerType,        /* void* f */
          llvmIntPtrType,         /* intptr_t gridXDim */
          llvmIntPtrType,         /* intptr_t gridyDim */
          llvmIntPtrType,         /* intptr_t gridZDim */
          llvmIntPtrType,         /* intptr_t blockXDim */
          llvmIntPtrType,         /* intptr_t blockYDim */
          llvmIntPtrType,         /* intptr_t blockZDim */
          llvmInt32Type,          /* unsigned int sharedMemBytes */
          llvmPointerType,        /* void *hstream */
          llvmPointerPointerType, /* void **kernelParams */
      }};
  FunctionCallBuilder rtPGOGetAlternativeCallBuilder = {
      "mgpurtPGOGetAlternative",
      llvmInt32Type,
      {
          llvmPointerType, /* const char *kernelId */
          llvmInt32Type,   /* int totalAlternatives */
      }};
  FunctionCallBuilder rtPGOStartCallBuilder = {
      "mgpurtPGOStart",
      llvmVoidType,
      {
          llvmPointerType, /* const char *kernelId */
          llvmInt32Type,   /* int totalAlternatives */
      }};
  FunctionCallBuilder rtPGOEndCallBuilder = {
      "mgpurtPGOEnd",
      llvmVoidType,
      {
          llvmPointerType, /* const char *kernelId */
          llvmInt32Type,   /* int totalAlternatives */
      }};
  FunctionCallBuilder streamCreateCallBuilder = {
      "mgpuStreamCreate", llvmPointerType /* void *stream */, {}};
  FunctionCallBuilder streamDestroyCallBuilder = {
      "mgpuStreamDestroy", llvmVoidType, {llvmPointerType /* void *stream */}};
  FunctionCallBuilder streamSynchronizeCallBuilder = {
      "mgpuStreamSynchronize",
      llvmVoidType,
      {llvmPointerType /* void *stream */}};
};

template <typename OpTy>
class ConvertOpToGpuRuntimeCallPattern : public ConvertOpToLLVMPattern<OpTy>,
                                         public GpuRuntimeCallBuilders {
public:
  explicit ConvertOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<OpTy>(typeConverter),
        GpuRuntimeCallBuilders(&typeConverter.getContext(), typeConverter) {}
};

} // namespace

#endif
