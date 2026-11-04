/*===============================================================*/
/*                                                               */
/*                            sgd.h                              */
/*                                                               */
/*          Top-level hardware function declaration              */
/*                                                               */
/*===============================================================*/

#include "../host/typedefs.h"

// top-level function declaration
#pragma SDS data mem_attribute(data:NON_CACHEABLE|PHYSICAL_CONTIGUOUS, label:CACHEABLE|PHYSICAL_CONTIGUOUS, theta:CACHEABLE|PHYSICAL_CONTIGUOUS)
#pragma SDS data data_mover(data:AXIDMA_SG, label:AXIFIFO, theta:AXIDMA_SG)
#pragma SDS data sys_port(data:AFI, label:AFI, theta:AFI)
#pragma SDS data zero_copy(theta[0:NUM_FEATURES / F_VECTOR_SIZE])
#pragma SDS data copy(data[0:NUM_FEATURES * NUM_TRAINING/D_VECTOR_SIZE])
#pragma SDS data zero_copy(label[0:NUM_TRAINING/L_VECTOR_SIZE])
#pragma SDS data access_pattern(data:SEQUENTIAL, label:SEQUENTIAL, theta:SEQUENTIAL) 
void SgdLR( VectorDataType    data[NUM_FEATURES * NUM_TRAINING / D_VECTOR_SIZE],
            VectorLabelType   label[NUM_TRAINING / L_VECTOR_SIZE],
            VectorFeatureType theta[NUM_FEATURES / F_VECTOR_SIZE],
            bool readLabels,
            bool writeOutput);

