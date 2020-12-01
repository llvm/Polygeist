#map = affine_map<()[s0] -> (s0)>

func @gemver(%alpha: f32, %beta: f32,
             %A: memref<?x?xf32>,
             %u1: memref<?xf32>,
             %v1: memref<?xf32>,
             %u2: memref<?xf32>,
             %v2: memref<?xf32>,
             %w: memref<?xf32>,
             %x: memref<?xf32>,
             %y: memref<?xf32>,
             %z: memref<?xf32>) {
  %c0 = constant 0 : index
  %N = dim %A, %c0 : memref<?x?xf32>

  affine.for %i = 0 to #map()[%N] {
    affine.for %j = 0 to #map()[%N] {
      %0 = affine.load %u1[%i] : memref<?xf32>
      %1 = affine.load %v1[%j] : memref<?xf32>
      %2 = mulf %0, %1 : f32
      %3 = affine.load %u2[%i] : memref<?xf32>
      %4 = affine.load %v2[%j] : memref<?xf32>
      %5 = mulf %3, %4 : f32
      %6 = affine.load %A[%i, %j] : memref<?x?xf32>
      %7 = addf %6, %2 : f32
      %8 = addf %7, %5 : f32
      affine.store %8, %A[%i, %j] : memref<?x?xf32>
    }
  }

  affine.for %i = 0 to #map()[%N] {
    affine.for %j = 0 to #map()[%N] {
      %9 = affine.load %A[%j, %i] : memref<?x?xf32>
      %10 = mulf %beta, %9 : f32
      %11 = affine.load %y[%j] : memref<?xf32>
      %12 = mulf %10, %11 : f32
      %13 = affine.load %x[%i] : memref<?xf32>
      %14 = addf %12, %13 : f32
      affine.store %14, %x[%i] : memref<?xf32>
    }
  }

  affine.for %i = 0 to #map()[%N] {
    %15 = affine.load %z[%i] : memref<?xf32>
    %16 = affine.load %x[%i] : memref<?xf32>
    %17 = addf %16, %15 : f32
    affine.store %17, %x[%i] : memref<?xf32>
  }

  affine.for %i = 0 to #map()[%N] {
    affine.for %j = 0 to #map()[%N] {
      %18 = affine.load %A[%i, %j] : memref<?x?xf32>
      %19 = mulf %alpha, %18 : f32
      %20 = affine.load %x[%j] : memref<?xf32>
      %21 = mulf %19, %20 : f32
      %22 = affine.load %w[%i] : memref<?xf32>
      %23 = addf %22, %21 : f32
      affine.store %23, %w[%i] : memref<?xf32>
    }
  }

  return
}

