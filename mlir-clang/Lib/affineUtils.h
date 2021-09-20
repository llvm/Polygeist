#ifndef MLIR_CLANG_AFFINE_UTILS_H
#define MLIR_CLANG_AFFINE_UTISL_H

namespace clang {
class VarDecl;
} // end namespace clang

namespace mlir {
class Value;
class Type;
} // end namespace mlir

namespace mlirclang {

class AffineLoopDescriptor {
private:
  mlir::Value upperBound;
  mlir::Value lowerBound;
  int64_t step;
  mlir::Type indVarType;
  clang::VarDecl *indVar;
  bool forwardMode;

public:
  AffineLoopDescriptor()
      : upperBound(nullptr), lowerBound(nullptr),
        step(std::numeric_limits<int64_t>::max()), indVarType(nullptr),
        indVar(nullptr), forwardMode(true){};
  AffineLoopDescriptor(const AffineLoopDescriptor &) = delete;

  auto getLowerBound() const { return lowerBound; }
  void setLowerBound(mlir::Value value) { lowerBound = value; }

  auto getUpperBound() const { return upperBound; }
  void setUpperBound(mlir::Value value) { upperBound = value; }

  int getStep() const { return step; }
  void setStep(int value) { step = value; };

  clang::VarDecl *getName() const { return indVar; }
  void setName(clang::VarDecl *value) { indVar = value; }

  mlir::Type getType() const { return indVarType; }
  void setType(mlir::Type type) { indVarType = type; }

  bool getForwardMode() const { return forwardMode; }
  void setForwardMode(bool value) { forwardMode = value; };
};

} // end namespace mlirclang

#endif
