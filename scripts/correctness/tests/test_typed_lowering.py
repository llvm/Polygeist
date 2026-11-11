from pathlib import Path
import sys
import types
import unittest
BASE=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(BASE/'scripts/correctness'),str(BASE/'scripts/correctness/lib_v2')]
import lower_v2 as L
class TypedLowering(unittest.TestCase):
    def test_small_integer_descriptors_verify(self):
        import subprocess,tempfile
        for elem,byte in [('i8',1),('i16',2)]:
            with self.subTest(elem=elem):
                e=L.Emitter('    ','typed');av=L.ArrayVal(e,'%input',f'memref<?x?x{elem}>',elem);av.layout(e,None,'input')
                text='module {\n  func.func @test(%input: memref<?x?x'+elem+'>) {\n'+'\n'.join(e.lines)+'\n    return\n  }\n}\n'
                result=subprocess.run([str(BASE/'build/bin/polygeist-opt')],input=text,text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(L.ELEM_BYTES[elem],byte)
    def test_element_width_mismatch_still_rejected(self):
        with self.assertRaises(L.LowerError):L.ArrayVal(L.Emitter('','bad'),'%input','memref<?xi8>','i16')
if __name__=='__main__':unittest.main()
