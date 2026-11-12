from pathlib import Path
import sys
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / 'scripts/correctness'))
import kernel_match_rewrite as km
base = Path(__file__).parent
cases = {
    'depthwise_conv3x3_cpu': 'cudnnDepthwiseConvolution2D_f32_memref',
    'flip_tensor_transform_cpu': 'cublasReverse1D_f32_memref',
    'mse_loss': 'cubMean1D_f32_memref',
    'nested_batch_offsets_cpu': 'cubExclusiveSum1D_i32_memref',
}
def rewrite(text, **options):
    return km.rewrite_mlir(text, enable_structured_rewrite=True, **options)[0]
for name, symbol in cases.items():
    source = (base / (name + '.mlir')).read_text()
    output = rewrite(source)
    assert 'kernel.launch @' + symbol in output, name
    if name == 'mse_loss':
        assert 'kernel.launch @cudnnPointwiseGraph_f32' in output
        wrongs = [source.replace('2.560000e+02', '2.550000e+02'),
                  source.replace('constant 0.000000e+00', 'constant 1.000000e+00')]
        limited = rewrite(source, max_launches=0)
        assert 'kernel.launch' not in limited and 'affine.store %cst_0' in limited
    elif name == 'nested_batch_offsets_cpu':
        wrongs = [source.replace('constant 0 : i32', 'constant 1 : i32'),
                  source.replace('(d0 + 1)', '(d0 + 2)'),
                  source.replace('arith.addi', 'arith.subi')]
        assert symbol in rewrite(source.replace('to 64 ', 'to 7 '))
    elif name == 'flip_tensor_transform_cpu':
        wrongs = [source.replace('-d1 + 63', 'd1'),
                  source.replace('-d0 + 31', '-d0 + 30')]
    else:
        wrongs = [source.replace('-d0 - d1 + 16', '-d0 - d1 + 15'),
                  source.replace('d3 + d1 - 1', 'd3 + d1'),
                  source.replace('memref<?x8x16x16xf32>', 'memref<?x?x16x16xf32>')]
        assert 'sizes: [1, 8, 16, 16]' in output
    for wrong in wrongs:
        assert wrong != source and 'kernel.launch @' + symbol not in rewrite(wrong), name
for tag in ['01', '10']:
    source = (base / ('log_sigmoid_' + tag + '.mlir')).read_text()
    symbol = 'kernel.launch @cudnnLogSigmoid_f32_memref'
    assert symbol in rewrite(source)
    wrong = source.replace('memref.copy %4, %arg2', 'memref.copy %4, %arg1') if tag == '01' else source.replace('memref.copy %4, %arg1', 'memref.copy %4, %arg2')
    assert wrong != source and symbol not in rewrite(wrong)
# SSA names are scoped to functions: a preceding zero seed cannot justify
# replacing a later nonzero-seeded sum with a mean.
source = (base / 'mse_loss.mlir').read_text()
start = source.index('  func.func ')
end = source.index('\n  }', start) + 4
body = source[start:end]
second = (body.replace('@renamed_kernel', '@nonzero_mean')
          .replace('affine.store %cst_0, %alloca[]', 'affine.store %one, %alloca[]')
          .replace('    %alloca =', '    %one = arith.constant 1.0 : f32\n    %alloca ='))
combined = source[:end] + '\n' + second + source[end:]
assert rewrite(combined).count('kernel.launch @cubMean1D_f32_memref') == 1
print('PASS: unnamed depthwise, reverse, mean and prefix routes; invalid domains rejected')
