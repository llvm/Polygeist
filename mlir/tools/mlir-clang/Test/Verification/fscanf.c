// RUN: mlir-clang %s | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main() {
    int no_of_nodes;

	scanf("%d",&no_of_nodes);
   
	// allocate host memory
	int* h_graph_nodes = (int*) malloc(sizeof(int)*no_of_nodes);

	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		scanf("%d\n", &h_graph_nodes[i]);
    }
}

// CHECK: func @main