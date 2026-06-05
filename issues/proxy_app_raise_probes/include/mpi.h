#ifndef PROXY_APP_RAISE_PROBE_MPI_H
#define PROXY_APP_RAISE_PROBE_MPI_H

typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Request;
typedef int MPI_Status;

#define MPI_COMM_WORLD 0
#define MPI_ORDER_C 0
#define MPI_DOUBLE_COMPLEX 0
#define MPI_REQUEST_NULL 0
#define MPI_STATUS_IGNORE ((MPI_Status *)0)

int MPI_Cart_sub(MPI_Comm comm, const int remaining_dims[], MPI_Comm *newcomm);
int MPI_Dims_create(int nnodes, int ndims, int dims[]);
int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[],
                    const int periods[], int reorder, MPI_Comm *comm_cart);
int MPI_Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[],
                 int coords[]);
int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]);
int MPI_Cart_rank(MPI_Comm comm, int coords[], int *rank);
int MPI_Barrier(MPI_Comm comm);
int MPI_Type_create_subarray(int ndims, const int sizes[], const int subsizes[],
                             const int starts[], int order, MPI_Datatype oldtype,
                             MPI_Datatype *newtype);
int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_commit(MPI_Datatype *datatype);
int MPI_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 int dest, int sendtag, void *recvbuf, int recvcount,
                 MPI_Datatype recvtype, int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status);
int MPI_Type_free(MPI_Datatype *datatype);
int MPI_Comm_free(MPI_Comm *comm);
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Wait(MPI_Request *request, MPI_Status *status);

#endif
