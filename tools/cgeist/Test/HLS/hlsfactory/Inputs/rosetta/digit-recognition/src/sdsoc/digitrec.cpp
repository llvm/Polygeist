/*===============================================================*/
/*                                                               */
/*                          digitrec.cpp                         */
/*                                                               */
/*             Hardware function for digit recognition           */
/*                                                               */
/*===============================================================*/

#include "../host/typedefs.h"

// popcount function
int popcount(WholeDigitType x)
{
  // most straightforward implementation
  // actually not bad on FPGA
  int cnt = 0;
  for (int i = 0; i < 256; i ++ )
    cnt = cnt + x[i];

  return cnt;
}

// Given the test instance and a (new) training instance, this
// function maintains/updates an array of K minimum
// distances per training set.
void update_knn( WholeDigitType test_inst, WholeDigitType train_inst, int min_distances[K_CONST] ) 
{
  #pragma HLS inline

  // Compute the difference using XOR
  WholeDigitType diff = test_inst ^ train_inst;

  int dist = 0;

  dist = popcount(diff);

  int max_dist = 0;
  int max_dist_id = K_CONST+1;
  int k = 0;

  // Find the max distance
  FIND_MAX_DIST: for ( int k = 0; k < K_CONST; ++k ) 
  {
    if ( min_distances[k] > max_dist ) 
    {
      max_dist = min_distances[k];
      max_dist_id = k;
    }
  }

  // Replace the entry with the max distance
  if ( dist < max_dist )
    min_distances[max_dist_id] = dist;

  return;
}

// Given 10xK minimum distance values, this function
// finds the actual K nearest neighbors and determines the
// final output based on the most common int represented by
// these nearest neighbors (i.e., a vote among KNNs).
LabelType knn_vote( int knn_set[PAR_FACTOR * K_CONST] ) 
{
  #pragma HLS inline

  // local buffers

  // final K nearest neighbors
  int min_distance_list[K_CONST];
  #pragma HLS array_partition variable=min_distance_list complete dim=0
  // labels for the K nearest neighbors
  int label_list[K_CONST];
  #pragma HLS array_partition variable=label_list complete dim=0
  // voting boxes
  int vote_list[10];
  #pragma HLS array_partition variable=vote_list complete dim=0

  int pos = 1000;

  // initialize
  INIT_1: for (int i = 0;i < K_CONST; i ++ )
  {
    #pragma HLS unroll
    min_distance_list[i] = 256;
    label_list[i] = 9;
  }

  INIT_2: for (int i = 0;i < 10; i ++ )
  {
    #pragma HLS unroll
    vote_list[i] = 0;
  }

  // go through all the lanes
  // do an insertion sort to keep a sorted neighbor list
  LANES: for (int i = 0; i < PAR_FACTOR; i ++ )
  {
    INSERTION_SORT_OUTER: for (int j = 0; j < K_CONST; j ++ )
    {
      #pragma HLS pipeline
      pos = 1000;
      INSERTION_SORT_INNER: for (int r = 0; r < K_CONST; r ++ )
      {
        #pragma HLS unroll
        pos = ((knn_set[i*K_CONST+j] < min_distance_list[r]) && (pos > K_CONST)) ? r : pos;
      }

      INSERT: for (int r = K_CONST ;r > 0; r -- )
      {
        #pragma HLS unroll
        if(r-1 > pos)
        {
          min_distance_list[r-1] = min_distance_list[r-2];
          label_list[r-1] = label_list[r-2];
        }
        else if (r-1 == pos)
        {
          min_distance_list[r-1] = knn_set[i*K_CONST+j];
          label_list[r-1] = i / (PAR_FACTOR / 10);
        }
      }
    }
  }

  // vote
  INCREMENT: for (int i = 0;i < K_CONST; i ++ )
  {
    #pragma HLS pipeline
    vote_list[label_list[i]] += 1;
  }

  LabelType max_vote;
  max_vote = 0;

  // find the maximum value
  VOTE: for (int i = 0;i < 10; i ++ )
  {
    #pragma HLS unroll
    if(vote_list[i] >= vote_list[max_vote])
    {
      max_vote = i;
    }
  }

  return max_vote;

}

// top-level hardware function
// since AXIDMA_SIMPLE interface does not support arrays with size more than 16384 on interface
// we call this function twice to transfer data
void DigitRec(WholeDigitType global_training_set[NUM_TRAINING / 2], WholeDigitType global_test_set[NUM_TEST], LabelType global_results[NUM_TEST], int run) 
{

  // This array stores K minimum distances per training set
  int knn_set[PAR_FACTOR * K_CONST];
  //#pragma HLS array_partition variable=knn_set complete dim=0

  static WholeDigitType training_set [NUM_TRAINING];
  // to be used in a pragma
  const int unroll_factor = PAR_FACTOR;
  //#pragma HLS array_partition variable=training_set block factor=unroll_factor dim=0

  static WholeDigitType test_set     [NUM_TEST];
  static LabelType results           [NUM_TEST];

  // the first time, just do data transfer and return
  if (run == 0)
  {
    // copy the training set for the first time
    lp1:for (int i = 0; i < NUM_TRAINING / 2; i ++ )
      //#pragma HLS pipeline
      training_set[i] = global_training_set[i];
    return;
  }

  // for the second time
  lp2:for (int i = 0; i < NUM_TRAINING / 2; i ++ )
    //#pragma HLS pipeline
    training_set[i + NUM_TRAINING / 2] = global_training_set[i];
  // copy the test set
  lp3:for (int i = 0; i < NUM_TEST; i ++ )
    //#pragma HLS pipeline
    test_set[i] = global_test_set[i];

  // loop through test set
  TEST_LOOP: for (int t = 0; t < NUM_TEST; ++t) 
  {
    // fetch one instance
    WholeDigitType test_instance = test_set[t];

    // Initialize the knn set
    SET_KNN_SET: for ( int i = 0; i < K_CONST * PAR_FACTOR ; ++i ) 
    {
      #pragma HLS unroll
      // Note that the max distance is 256
      knn_set[i] = 256;
    }

    TRAINING_LOOP : for ( int i = 0; i < NUM_TRAINING / PAR_FACTOR; ++i ) 
    {
      //#pragma HLS pipeline
      LANES : for ( int j = 0; j < PAR_FACTOR; j++ ) 
      {
        //#pragma HLS unroll
        // Read a new instance from the training set
        WholeDigitType training_instance = training_set[j * NUM_TRAINING / PAR_FACTOR + i];

        // Update the KNN set
        update_knn( test_instance, training_instance, &knn_set[j * K_CONST] );
      }
    }
    // Compute the final output
    LabelType max_vote = knn_vote(knn_set);
    results[t] = max_vote;

  }

  // copy the results out
  lp4: for (int i = 0; i < NUM_TEST; i ++ )
    //#pragma HLS pipeline
    global_results[i] = results[i];

}

