from pathlib import Path
import sys
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'scripts/correctness'))
import kernel_match_rewrite as km
base=Path(__file__).parent
for source in sorted(base.glob('*.mlir')):
    original=source.read_text()
    def selected(text):
        return 'kernel.launch @cubSegmentedValue' in km.rewrite_mlir(text,enable_structured_rewrite=True)[0]
    assert selected(original),source
    assert selected(original.replace('64','7').replace('63','6')),(source,'width')
    assert not selected(original.replace('cmpf ogt','cmpf oge').replace('cmpf olt','cmpf ole')),(source,'last tie')
    assert not selected(original.replace('cmpf ogt','cmpf ugt').replace('cmpf olt','cmpf ult')),(source,'unordered')
    wrong=(original.replace('affine.load %arg0[0]','affine.load %arg0[1]')
           if '_all_' in source.name else original.replace('%arg0[0, 0]','%arg0[0, 1]'))
    assert wrong!=original and not selected(wrong),(source,'wrong seed')
print('PASS: ordered first-value reductions, renamed inputs, width and negative checks')
