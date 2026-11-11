"""Whole entry wrappers can preserve real callees through the common recipe."""
import subprocess,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
class RecipeScope(unittest.TestCase):
 def test_dependency_scope_does_not_drop_callee_computation(self):
  source="""module {
 func.func @helper(%a: memref<4xf32>) {
  %v = arith.constant 1.0 : f32
  affine.for %i = 0 to 4 {
   affine.store %v, %a[%i] : memref<4xf32>
  }
  return
 }
 func.func @entry(%a: memref<4xf32>) {
  func.call @helper(%a) : (memref<4xf32>) -> ()
  return
 }
}
"""
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);(d/'in.mlir').write_text(source);results=[]
   for extra in [[],['--keep-dependencies']]:
    cmd=[sys.executable,str(HERE/'lib_v2/recipe.py'),'--stage','raise','--function','entry','--input',str(d/'in.mlir'),'--output',str(d/'out.mlir'),*extra]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=60);self.assertEqual(r.returncode,0,r.stdout+r.stderr);results.append((d/'out.mlir').read_text())
   self.assertNotIn('linalg.generic',results[0]);self.assertIn('linalg.generic',results[1]);self.assertIn('call @helper',results[1])
if __name__=='__main__':unittest.main()
