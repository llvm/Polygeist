from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/"lib_v2")]
import subprocess
import unittest
import c_types as T

class ExactStorageTypes(unittest.TestCase):
    def test_preprocessed_fixed_width_arrays(self):
        text=subprocess.check_output(['gcc','-E','-P','-x','c','-'],input='#include <stdint.h>\n',text=True)
        for parameter,expected in [('const uint8_t image[][w]',('image','unsigned char',['','w'])),('int16_t output[n]',('output','short',['n'])),('uint32_t bins[256]',('bins','unsigned int',['256']))]:
            self.assertEqual(T.parse_parameter(parameter,text),expected)

    def test_typedef_chain_preserves_unsigned_storage(self):
        text='typedef unsigned short storage; typedef storage pixel;'
        self.assertEqual(T.parse_parameter('const pixel input[]',text),('input','unsigned short',['']))

    def test_float_and_existing_dimension_spellings(self):
        self.assertEqual(T.parse_parameter('float alpha'),('alpha','float',[]))
        self.assertEqual(T.parse_parameter('long int n'),('n','long',[]))
        self.assertEqual(T.parse_parameter('double A[][lda]'),('A','double',['','lda']))

    def test_ambiguous_or_effectful_storage_rejected(self):
        for parameter in ['char image[]','volatile int input[]','long double x[]','struct pixel input[]','int *input']:
            with self.subTest(parameter=parameter),self.assertRaises(ValueError):T.parse_parameter(parameter)

    def test_pointer_alias_is_not_a_scalar_element(self):
        with self.assertRaises(ValueError):T.parse_parameter('ptr data[]','typedef int * ptr;')

if __name__=='__main__':unittest.main()
