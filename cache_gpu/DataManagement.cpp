//
//  DataManagement.cpp
//  Coordinate_descent
//
//  Created by Zhiwei Fan on 10/2/15.
//  Copyright (c) 2015 Zhiwei Fan. All rights reserved.
//


#include "DataManagement.h"
#include <iostream>
#include <fstream>
#include <sstream>


/*
 Notes for later work:
 minor revision: key should be kept int, however, use double here first for convenience
 */

DataManagement::DataManagement(){};

/**
 Note:
 Currently the file input is assumed have the table id simply as array index
 All files as input are assumed to be preprocessed, corresponding processing functionality to be added
 */

/**
 Function to be added (Not for research purpose):
 (Simple)
 1. Read the "actual table name" 
 2. Read the "actual name" of features from the table and store in the information file
 3. Read the "actual id" of features from the table and store in the information file
 */

/**
 Store each coloumn in the the file to be a "binary array" (table in column store)
 *Note: Table type, number of features and number of rows need to be known ahead
 @param fileName:the name of the file storing the table to be stored
 @param featurenum: number of features in the table
 @param table_type: 0 if the table to be stored is entity table S; 1 if the table to be stored is attribute table R; 2 if the table to be stored is  "Full table" T (tid,label,t_1,t_2,t_3 ...)
 @param row_num: the number of rows in the table (number of lines in the input file)
 @corresponding file format for table S: sid,fk,y(label),x_s[](feature vector)
 @corresponding file format for table R: rid,x_r[](feature vector)
 Note: All entries in the tables are assumed to be numeric
 */
void DataManagement::store(string fileName, int feature_num, int table_type, long row_num)
{
    string table_name;
    vector<string> *table_field;
    int col_num = 0;
    
    // Decide the number of fields according to the table type and store the corresponding name of each field
    if(table_type == 0)
    {
        message("The table to be stored is entity table S");
        // sid + label + fk + x_s[]
        col_num = 3 + feature_num;
        table_name = "S";
        table_field = new vector<string>(col_num);
        table_field->at(0) = "sid";
        table_field->at(1) = "label";
        table_field->at(2) = "fk";
        for(int i = 0; i < feature_num; i ++)
        {
            table_field->at(3+i) = "x_s" + to_string((long long unsigned int)i);
        }
    }
    else if(table_type == 1){
        message("The table to be stores is attribute table R");
        // rid + x_r
        col_num = 1 + feature_num;
        table_name = "R";
        table_field = new vector<string>(col_num);
        table_field->at(0) = "rid";
        for(int i = 0; i < feature_num; i ++)
        {
            table_field->at(1+i) = "x_r" + to_string((long long unsigned int)i);
        }
    }
    else if(table_type == 2)
    {
        message("The table to be stores is Full table T");
        col_num = 2 + feature_num;
        table_name = "T";
        table_field = new vector<string>(col_num);
        table_field->at(0) = "tid";
        table_field->at(1) = "label";
        for(int i = 0; i < feature_num; i ++)
        {
            table_field->at(2+i) = "x_t" + to_string((long long unsigned int)i);
        }
    }
    else{
        errorMessage("Invalid table type identifier given");
        exit(1);
    }
    
    // Reading the file and load the data to the corresponding columns (binary array)
    message("Open the input file");
    printf("File Name: %s\n", fileName.c_str());
    ifstream infile;
    infile.open(fileName);
    
    // Memory free case: all columns can fit in memory the same time
    if(!infile.is_open())
    {
        errorMessage("Cannot open the file: the name of the file might be wrong or the file might be broken");
        exit(1);
    }
    
    message("Open the output File");
    // Open the corresponding number of files (representing columns) to store the loading data
    ofstream *outFile = new ofstream[col_num];
    // output buffer
    double *write =  new double[1];
    for(int i = 0; i < col_num; i ++)
    {
        outFile[i].open(table_name + "_" + table_field->at(i), ios::out | ios::binary);
        if(!outFile[i].is_open())
        {
            errorMessage("Cannot open file.");
            exit(1);
        }
    }
    
    message("Starting to load the data from the file into the database");
    // All valid files should seperate the columns with ' ' (one space)
    char delim = ' ';
    while(!infile.eof())
    {
        string s;
        getline(infile, s);
        vector<string> tuple;
        tuple = split(s,delim);
        //Skip the empty line
        if(tuple.size() == 0)
        {
            continue;
        }
        
        // Check the consistency of feature number
        if(tuple.size() != col_num )
        {
            message("Inconsistency of feature number");
        }
    
        // Write the value in each filed of current tuple into the corresponding columns
        for(int i = 0; i < col_num; i ++)
        {
            const char *cstr = tuple.at(i).c_str();
            write[0] = atof(cstr);
            outFile[i].write((char *) write, (sizeof(double)));
        }
    }
    infile.close();
    
    // Write the table information to a single file
    ofstream info(table_name + "_" + "info", ios::out | ios::app);
    if(!info.is_open())
    {
        errorMessage("Cannot create info file.");
        exit(1);
    }

    info<<"table name: "<<table_name<<endl;
    info<<"table type: "<<table_type<<endl;
    info<<"feature num: " <<feature_num<<endl;
    info<<"row number: "<<row_num<<endl;
    info.close();
    
   
    message("Finish loading");
    
    // Close all writing files
    for(int i = 0; i < col_num; i ++)
    {
        outFile[i].close();
    }
    
    // Relese all allocated memory
    delete [] write;
    delete [] outFile;
    delete table_field;
}

/**
 Read the content of a single column specified by the given column name
 @param fileName: the column name
 @param row_num: the number of entries in the column
 */
void DataManagement::readColumn(string fileName, long row_num)
{
    double *col = new double[row_num];
    ifstream inFile(fileName, ios::in | ios::binary);
    if(!inFile.is_open())
    {
        errorMessage("Cannot open file.");
        exit(1);
    }
    inFile.read((char *) col, row_num * (sizeof(double)));
    inFile.close();
    
    for(int i = 0; i < row_num; i ++)
    {
        printf("%f\n", col[i]);
    }
    
    delete [] col;
}

void DataManagement::fetchColumn(string fileName, long row_num, double *col )
{
    ifstream inFile(fileName, ios::in | ios::binary);
    if(!inFile.is_open())
    {
        errorMessage("Cannot open file.");
        exit(1);
    }
    inFile.read((char *) col, row_num * (sizeof(double)));
    inFile.close();
}

/**
 Get the name of the fields of the table specified
 @param fileName: corresponding info file containing the information of the table
 @param fields: reference for the vector used to store the name of corresponding fields
 tableInfo:table_type, table_feature_num, table_row_num
 */
vector<string> DataManagement::getFieldNames(string tableName, vector<long> &tableInfo)
{
    ifstream info;
    vector<string> fields;
    vector<string> table_info;
    int table_type;
    string table_name;
    int table_feature_num;
    long table_row_num;
    
    info.open(tableName+"_info");
    if(!info.is_open())
    {
        errorMessage("Unable to read the given information file");
        exit(1);
    }
    
    string s;
    while(!info.eof())
    {
        getline(info,s);
        char delim = ' ';
        vector<string> tokens = split(s,delim);
        // Skip the empty line
        if(tokens.size() == 0)
        {
            continue;
        }
        table_info.push_back(tokens[2]);
    }
   
    printf("size of table info: %lu\n", table_info.size());
    table_name = table_info[0];
    table_type = atoi(table_info[1].c_str());
    table_feature_num = atoi(table_info[2].c_str());
    table_row_num = atol(table_info[3].c_str());
    
    message("Check the vector size to store the tableInfo: ");
    if(tableInfo.size() != 3)
    {
        errorMessage("The size of the vector to store the table info is wrong");
        exit(1);
    }
    tableInfo[0] = table_type;
    tableInfo[1] = table_feature_num;
    tableInfo[2] = table_row_num;
    
    message("Fetch the information of fields of the corresponding table");
    if(table_type == 0)
    {
        // Entity table S: sid + label + fk + x_s[]
        fields.push_back(table_name+"_"+"sid");
        fields.push_back(table_name+"_"+"label");
        fields.push_back(table_name+"_"+"fk");
        for(int i = 0; i < table_feature_num; i ++)
        {
            fields.push_back(table_name+"_"+"x"+"_s"+to_string((long long unsigned int)i));
        }
    }
    else if(table_type == 1)
    {
        // Entity table R: rid + x_r[]
        fields.push_back(table_name+"_"+"rid");
        for(int i = 0; i < table_feature_num; i ++)
        {
            fields.push_back(table_name+"_"+"x"+"_r"+to_string((long long unsigned int)i));
        }
    }
    else if(table_type == 2)
    {
        // Materialized table T: tid(sid) + label + x_s[] + x_r[]
        fields.push_back(table_name + "_" + "tid");
        fields.push_back(table_name + "_" + "label");
        for(int i = 0; i < table_feature_num; i ++)
        {
            fields.push_back(table_name + "_" + "x" + "_t" + to_string((long long unsigned int)i));
        }
    }
    else
    {
        errorMessage("Invalid table type: the information file is invalid");
        exit(1);
    }
    
    return fields;
}

/**
 Join the entity table S and attribute table R and return the "materialized" table
 @param table_name1: suppose to be entity table S
 @param table_name2: suppose to be attribute table R (More generalized version will be handled later)
 @param joinTable: The join
 */
void DataManagement::join(string table_name1, string table_name2, string joinTable)
{
   
    // Get the table information
    vector<string> table1_fields;
    vector<string> table2_fields;
    vector<long> table1_info(3);
    vector<long> table2_info(3);
    //Get the column names of two tables
    table1_fields = getFieldNames(table_name1, table1_info);
    table2_fields = getFieldNames(table_name2, table2_info);
    int table1_feature_num = (int)table1_info[1];
    long table1_row_num = table1_info[2];
    int table2_feature_num = (int)table2_info[1];
    long table2_row_num = table2_info[2];
    
    message("Start fetching KKMR reference");
    // OID-OID Mapping (Key Foreign-Key Mapping Reference)
    double *KKMR = new double[table1_row_num];
    // Read the fk column(referred rid in R) in table S, rid column in R
    ifstream fk;
    // Load the fk to KKMR
    fk.open(table1_fields[2], ios::in | ios::binary);
    // rid.open(table2_fields[0], ios::in | ios::binary);
    if(!fk.is_open())
    {
        errorMessage("Error Message: Cannot load the fk column.\n");
        exit(1);
    }
    fk.read((char *)KKMR, table1_row_num*(sizeof(double)));
    fk.close();
    message("Finish fetchig KKMR reference");
    
    message("Start fetching the column names for materialized (join) table T");
    // Get the features for table T: sid, label, x_s[], x_r[]
    int t_column_num = 1 + 1 + table1_feature_num + table2_feature_num;
    // Store the name of every column of table T
    vector<string> t_columns(t_column_num);
    // Same as sid(tid)
    t_columns[0] = joinTable + "_tid";
    t_columns[1] = joinTable + "_label";
    // Columns corresponding to x_s[]
    for(int i = 2; i < 2+table1_feature_num; i ++)
    {
        t_columns[i] = joinTable + "_" + "x_t" + to_string((unsigned long long)i-2);
    }
    // Columns corresponding to x_r[]
    for(int k = 2+table1_feature_num; k < t_column_num; k ++)
    {
        t_columns[k] = joinTable + "_" + "x_t" + to_string((unsigned long long)k-2);
    }
    message("Finish fetching the column names for materialized (join) table T");
    
    message("Open the output file for T");
    // Open the corresponding number of files (representing columns) for T
    ofstream *T = new ofstream[t_column_num];
    for(int i = 0; i < t_column_num; i ++)
    {
        T[i].open(t_columns[i], ios::out | ios::binary);
        if(!T[i].is_open())
        {
            printf("T[%d]:\n", i);
            errorMessage("Error Message: Cannot open file.");
            exit(1);
        }
    }
    
    message("Start writing sid,label to T");
    // First write corresponding columns sid,label in S
    ifstream sid;
    double *buffer = new double[table1_row_num];
    sid.open(table1_fields[0], ios::in | ios::binary);
    if(!sid.is_open())
    {
        errorMessage("Error Message: Cannot open file sid.");
        exit(1);
    }
    sid.read((char *)buffer, table1_row_num*(sizeof(double)));
    sid.close();
    T[0].write((char *)buffer, table1_row_num*(sizeof(double)));
    T[0].close();

    ifstream label;
    label.open(table1_fields[1], ios::in | ios::binary);
    if(!label.is_open())
    {
        errorMessage("Error Message: Cannot open file label");
        exit(1);
    }
    label.read((char *)buffer, table1_row_num*(sizeof(double)));
    label.close();
    T[1].write((char *)buffer, table1_row_num*(sizeof(double)));
    T[1].close();
    message("Finish writing sid,label to T");
    
    message("Start writing x_s[] to T");
    // Then write corresponding columns x_s[] in S
    for(int i = 3; i < table1_feature_num+3; i ++)
    {
        ifstream x_s;
        x_s.open(table1_fields[i], ios::in | ios::binary);
        if(!x_s.is_open())
        {
            errorMessage("Error Message: Cannot open file.");
            exit(1);
        }
        x_s.read((char *)buffer, table1_row_num*(sizeof(double)));
        x_s.close();
        T[i-1].write((char *)buffer, table1_row_num*(sizeof(double)));
        T[i-1].close();
    }
    message("Finish writing x_s[] to T");
    
    message("Start writing x_r[] to T");
    // Then write corresponding columns x_r[] in R using KKMR as reference
    // Buffer for reading and probing
    double *read = new double[table2_row_num];
    for(int k = 1; k <= table2_feature_num; k ++ )
    {
        ifstream x_r;
        x_r.open(table2_fields[k], ios::in | ios::binary);
        
        if(!x_r.is_open())
        {
            errorMessage("Error Message: Cannot open file.");
            exit(1);
        }
        
        x_r.read((char *) read, table2_row_num*(sizeof(double)));
        x_r.close();
        for(long m = 0; m < table1_row_num; m ++)
        {
            long fk = KKMR[m];
            buffer[m] = read[fk-1];
            
        }
        
        T[1+table1_feature_num+k].write((char *)buffer,table1_row_num*(sizeof(double)));
        T[1+table1_feature_num+k].close();
    }
    message("Finish writing x_r[] to T");
    
            message("Start writing the infomation file for table T");
    // Write the table information to a single file
    ofstream info(joinTable + "_" + "info", ios::out | ios::app);
    info<<"table name: "<<joinTable<<endl;
    info<<"table type: "<<2<<endl;
    info<<"feature num: "<<table1_feature_num + table2_feature_num<<endl;
    info<<"row number: "<<table1_row_num<<endl;
    info.close();

    
    delete [] KKMR;
    delete [] T;
    delete [] read;
    delete [] buffer;
}


/**
 Output the table in text format, could be used to validate the correctless of data loading
 @param tableName: the name of the table to be read
 */
void DataManagement::readTable(string tableName)
{
    message("Start reading the table");
    vector<long> tableInfo(3);
    vector<string> fields = getFieldNames(tableName,tableInfo);
    int col_num = (int)fields.size();
    long row_num = tableInfo[2];
    ifstream *inFile = new ifstream[col_num];
    // Input buffer
    double *read =  new double[1];
    
    message("Start loading the table: ");
    for(int i = 0; i < col_num; i ++)
    {
       
        string column = fields.at(i);
        printf("Current column: %s\n", column.c_str());
        inFile[i].open(column, ios::in | ios::binary);
        if(!inFile[i].is_open())
        {
            errorMessage("Cannot read the given table, the table may not exist.");
            exit(1);
        }
    }
    
    // Starting to read the table
    for(long j = 0; j < row_num; j ++)
    {
        for(int k = 0; k < col_num; k ++)
        {
	    inFile[k].read((char *)read, (sizeof (double)));
            if(k == 0)	
	    {
            	printf("%d ", (int)read[0]);
	    }
	    else
	    {
		printf("%lf ",read[0]);
	    }
        }
        printf("\n");
    }
    
    // Close all files
    for(int i = 0; i < col_num; i ++)
    {
        
        inFile[i].close();
    }
    
    delete [] inFile;
    delete [] read;
}

/**
 Parsing function
 */
vector<string> DataManagement::split(const string&s, char delim)
{
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while(getline(ss,item, delim))
    {
        tokens.push_back(item);
    }
    return tokens;
}

void DataManagement::message(const char *message)
{
    printf("%s\n",message);
}

void DataManagement::errorMessage(const char *error)
{
    fprintf(stderr, "%s\n", error);
}


// Row Store (Initially for Machine Learning Project: SGD, BGD)
vector< vector<double> > DataManagement::rowStore(string fileName)
{
    vector< vector<double> > data;
    
    ifstream infile;
    infile.open(fileName);
    
    if(!infile.is_open())
    {
        errorMessage("Cannot open the file: the name of the file might be wrong or the file might be broken");
        exit(1);
    }
    
    char delim = ' ';
    while(!infile.eof())
    {
        string s;
        getline(infile,s);
        vector<string> tuple;
        tuple = split(s, delim);
        
        vector<string> tokens = split(s,delim);
        int size = (int)tuple.size();
        if(tuple.size() == 0)
        {
            continue;
        }
        
        vector<double> temp;
        for(int i = 0; i < size; i ++)
        {
            double cur_entry = atof(tuple.at(i).c_str());
            temp.push_back(cur_entry);
        }
        
        data.push_back(temp);
    }
    
    return data;
}

