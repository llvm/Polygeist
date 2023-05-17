#include <stdio.h>
#include <stdlib.h>
#include <libpq-fe.h>

// PGresult *PQexec(PGconn*, const char* command);
// PQgetvalue
//    %7 = call @PQexec(%2, %6) : (memref<?x1xi8>, memref<?xi8>) -> memref<?x1xi8>
#pragma lower_to(num_rows_fn, "sql.num_results")
int num_rows_fn(size_t);// char*

void do_exit(PGconn *conn) {
    
    PQfinish(conn);
    exit(1);
}

int main() {
    
    PGconn *conn = PQconnectdb("user=janbodnar dbname=testdb");

    if (PQstatus(conn) == CONNECTION_BAD) {
        
        fprintf(stderr, "Connection to database failed: %s\n",
            PQerrorMessage(conn));
        do_exit(conn);
    }

    PGresult *res = PQexec(conn, "SELECT VERSION()");    
    
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {

        printf("No data retrieved\n");        
        PQclear(res);
        do_exit(conn);
    }    

    printf("%s\n", PQgetvalue(res, 0, 0));
    printf("%d\n", num_rows_fn((size_t)res));
    // res, 0, 0));

    PQclear(res);
    PQfinish(conn);

    return 0;
}
