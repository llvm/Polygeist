"""The source validator compares all integer bits, including 64-bit extrema."""
from pathlib import Path
import sys, subprocess, tempfile, unittest
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'lib_v2'))
import validate_specs as V


class IntegerValidation(unittest.TestCase):
    def test_exact_integer_comparisons_and_source_types(self):
        with tempfile.TemporaryDirectory() as directory:
            D = Path(directory)
            results=[]
            for ct in ['int8_t','uint8_t','int16_t','uint16_t','int32_t','uint32_t','int64_t','uint64_t']:
             elem,signed,bits=V.INTEGER_TYPES[ct]
             params=V.c_param_kinds('probe',[('%a',f'memref<256x{elem}>')],['a'],set(),{'a':ct})
             for incorrect in [False,True]:
              source=V.harness('probe',params,None)
              source+=f'\nvoid probe_ref({ct} *a) {{}}\nvoid probe_impl({ct} *allocated, {ct} *a, int64_t offset, int64_t size, int64_t stride) {{'+('a[1] -= 1;' if incorrect else '')+'}\n'
              path=D/f'{ct}_{incorrect}.c';path.write_text(source);binary=path.with_suffix('')
              subprocess.run(['cc','-O0',str(path),'-lm','-o',str(binary)],check=True,capture_output=True)
              p=subprocess.run([str(binary)],text=True,capture_output=True)
              assert p.returncode==int(incorrect),(ct,incorrect,p.stdout)
              results.append(dict(dtype=ct,perturbed=incorrect,status='PASS',output=p.stdout.strip()))
            for elem,ctype in [('i16','uint8_t'),('i16','unknown_t')]:
             try:V.c_param_kinds('probe',[('%a',f'memref<?x{elem}>')],['a'],set(),{'a':ctype})
             except ValueError:results.append(dict(dtype=ctype,status='REJECTED'))
             else:raise AssertionError('unsafe integer source type accepted')


if __name__ == '__main__':
    unittest.main()
