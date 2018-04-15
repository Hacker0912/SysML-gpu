//
//  DataManagement.h
//  Coordinate_descent
//
//  Created by Zhiwei Fan on 10/2/15.
//  Copyright (c) 2015 Zhiwei Fan. All rights reserved.
//

#ifndef __Coordinate_descent__DataManagement__
#define __Coordinate_descent__DataManagement__

using namespace std;
#include <vector>
#include <stdio.h>
#include <cstdlib>
#include <string>

// Read data from table stored in text file and loaded into the DB
class DataManagement
{
public:
    DataManagement();
    
    void store(string FileName, int feature_num, int table_type, long row_num);
    void readColumn(string fileName, long row_num);
    void fetchColumn(string fileName, long row_num, double *col);
    void join(string table_name1, string table_name2, string joinTable);
    vector<string> getFieldNames(string tableName,vector<long> &tableInfo);
    void readTable(string tableName);
    static void message(const char *message);
    static void errorMessage(const char *error);
    vector< vector<double> > rowStore(string fileName);
private:
    vector<string> split(const string &s, char delim);
};

#endif /* defined(__Coordinate_descent__DataManagement__) */
