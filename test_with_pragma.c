#include <libpq-fe.h>
#include <stdio.h>
#include <stdlib.h>

// PGresult *PQexec(PGconn*, const char* command);
// PQgetvalue
//    %7 = call @PQexec(%2, %6) : (memref<?x1xi8>, memref<?xi8>) ->
//    memref<?x1xi8>
// #pragma lower_to(num_rows_fn, "sql.num_results")
// int num_rows_fn(size_t);// char*

// #pragma lower_to(get_value_fn_int, "sql.get_value")
// int get_value_fn_int(size_t, int, int);

// #pragma lower_to(get_value_fn_double, "sql.get_value")
// double get_value_fn_double(size_t, int, int);

// #pragma lower_to(execute, "sql.execute")
// PGresult* execute(size_t, char*);

void do_exit(PGconn *conn) {
  PQfinish(conn);
  exit(1);
}

int main() {

  PGconn *conn = PQconnectdb("user=carl dbname=testdb");

  if (PQstatus(conn) == CONNECTION_BAD) {

    fprintf(stderr, "Connection to database failed: %s\n",
            PQerrorMessage(conn));
    do_exit(conn);
  }

  // PGresult *res = PQexec(conn, "SELECT 17");
  PGresult *res = PQexec(conn, "SELECT a FROM table1");
  PGresult *res1 = PQexec(conn, "SELECT * FROM table1 WHERE b > 10 OR c < 10 AND a <= 20");
  PGresult *res2 = PQexec(conn, "SELECT * FROM table1 WHERE b > 10 AND c < 10");
  PGresult *res3 = PQexec(conn, "SELECT b, c FROM table1 WHERE a <= 10 LIMIT 10");
  // PGresult *res3 = PQexec(conn, "SELECT b, c FROM table1 LIMIT ALL");
  if (PQresultStatus(res) != PGRES_TUPLES_OK) {

    printf("No data retrieved\n");
    PQclear(res);
    do_exit(conn);
  }

  PQclear(res);
  PQclear(res1);
  PQclear(res2);
  PQclear(res3);
  PQfinish(conn);

  return 0;
}
