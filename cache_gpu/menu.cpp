/*
 Display the menu 
*/

#include<stdio.h>

int main()
{
    printf("options:\n");
    printf("Type 0: S; Type 1: R; Type 2: T\n");
    // Argument number: 6
    printf("1. Create table: DB create <file_name(table_name)> <type> <feature_num row_num>\n");
    // Argument number: 5
    printf("2. Join table: DB join <table_name_S> <table_name_R> <joinTable_name>\n");
    // Argument number: 3
    printf("3. Read table: DB read <table_name>\n");
    // Argument number: 4
    printf("4. Read Column: DB readColumn <column_name> <row_num>\n");
    // Argument number: 8
    printf("5. Materialize SCD: DB m <table_name> <step_size> <error_tolerance> <iter_num> <linear model> <avail_mem>\n");
    // Argument number: 9
    printf("6. Stream SCD: DB s <table_name_S> <table_name_R> <step_size> <error_tolerance> <iter_num> <linear model> <avail_mem>\n");
    // Argument number: 9
    printf("7. Factorize SCD: DB f <table_name_S> <table_name_R> <step_size> <error_tolerance> <iter_num> <linear model> <avail_mem>\n");
    // Argument number: 9
    printf("8. Materialize BCD: DB BCD <table_name> <block_size> <step_size> <error_tolerance> <iter_num> <linear model> <avail_mem>\n");
    // Argument number: 10
    printf("9. Factorize fBCD: DB fBCD <table_name_S> <table_name_R> <block_size> <step_size> <error_tolerance> <iter_num> <linear model> <avail_mem>\n");
    // Argument number: 4
    printf("10. SGD: DB SGD <train_file_name> <test_file_name>\n");
    // Argument number: 9
    printf("11. BGD: DB BGD <train_file_name> <test_file_name>\n");
    return 0;
}
