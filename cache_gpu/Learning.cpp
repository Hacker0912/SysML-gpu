//
//  main.cpp
//  Coordinate_descent
//
//  Created by Zhiwei Fan on 10/2/15.
//  Copyright (c) 2015 Zhiwei Fan. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include "DataManagement.h"
#include "techniques.h"

int main(int argc, const char * argv[])
{
    DataManagement dM;
    techniques t;
    string option1 = "create";
    string option2 = "join";
    string option3 = "read";
    string option4 = "readColumn";
    string option5 = "m";
    string option6 = "s";
    string option7 = "f";
    string option8 = "BCD";
    string option9 = "fBCD";
    string option10 = "SGD";
    string option11 = "BGD";
   
    if(argc == 6 && argv[1] == option1)
    {
        // option 1: create table from the corresponding textfile
        string fileName = argv[2];
        int tableType = atoi(argv[3]);
        int featureNum = atoi(argv[4]);
        long rowNum = atol(argv[5]);
        dM.store(fileName, featureNum, tableType, rowNum);
    }
    else if(argc == 5 && argv[1] == option2)
    {
        // option 2: join two tables S and R -> table T
        string tableName_S = argv[2];
        string tableName_R = argv[3];
        string joinTableName = argv[4];
        dM.join(tableName_S, tableName_R, joinTableName);
    }
    else if(argc == 3 && argv[1] == option3)
    {
        // option 3: read table
        string tableName = argv[2];
        dM.readTable(tableName);
    }
    else if(argc == 4)
    {
        string option = argv[1];
        if(option == option4)
        {
            // option 4: read column
            string column_name = argv[2];
            long row_num = atol(argv[3]);
            dM.readColumn(column_name, row_num);
        }
        else
        {
            if( (option != option10) && (option != option11))
            {
                dM.errorMessage("Invalid option or wrong arguments");
                exit(1);
            }
            else
            {
                string tableName = argv[2];
                string testTable = argv[3];
                double *model;
                int feature_num = 0;
                vector<double> model_vector;
                setting _setting;
                printf("Setting stepSize: \n");
                if(1 != scanf("%lf",&_setting.step_size)) {
                    printf("Error reading in stepSize\n");
                    exit(1);
                }
                printf("Setting error tolearence: \n");
                if(1 != scanf("%lf",&_setting.error)) {
                    printf("Error reading in tolearance\n");
                    exit(1);
                }
                printf("Setting number of iterations: \n");
                if(1 != scanf("%d",&_setting.iter_num)) {
                    printf("Error reading in iteration number\n");
                    exit(1);
                }
                   
                if(option == option10)
                {
                    // option 10: SGD
                    vector< vector<double> > data = dM.rowStore(tableName);
                    int featureNum = (int)(data.at(0).size()-2);
                    feature_num = featureNum;
                    t.SGD(data, _setting, model, featureNum);
                }
                else
                {
                    // option 11: BGD
                    vector< vector<double> > data = dM.rowStore(tableName);
                    int featureNum = (int)(data.at(0).size()-2);
                    feature_num = featureNum;
                    t.BGD(data, _setting, model, featureNum);
                }
                   
                vector< vector<double> > testData = dM.rowStore(testTable);
                for(int i = 0; i < feature_num; i ++)
                {
                    model_vector.push_back(model[i]);
                }
                t.classify(testData, model_vector);
                   
                delete [] model;
            }
        }
    }
    else if(argc == 8 && argv[1] == option5)
    {
      
        // option 5: Materialize SCD
        string tableName = argv[2];
        double step_size = atof(argv[3]);
        double error_tolerance = atof(argv[4]);
        int iter_num = atoi(argv[5]);
        const char *lm = argv[6];
        double avail_mem = atof(argv[7]);
        double *model;
        
        // Setting
        setting _setting;
        _setting.step_size = step_size;
        _setting.error = error_tolerance;
        _setting.iter_num = iter_num;

        t.materialize(tableName, _setting, model, avail_mem, lm);
        delete [] model;
    }
    else if(argc == 9)
    {
        string option = argv[1];
        
        if(option == option6 || option == option7)
        {
            string tableName_S = argv[2];
            string tableName_R = argv[3];
            double step_size = atof(argv[4]);
            double error_tolerance = atof(argv[5]);
            int iter_num = atoi(argv[6]);
            const char *lm = argv[7];
            double avail_mem = atof(argv[8]);
            double *model;
            
            // Setting
            setting _setting;
            _setting.step_size = step_size;
            _setting.error = error_tolerance;
            _setting.iter_num = iter_num;
            
            if(option == option6)
            {
                // option 6: Stream SCD
                t.stream(tableName_S, tableName_R, _setting, model, avail_mem, lm);
            }
            else
            {
                // option 7: Factorize SCD
                t.factorize(tableName_S, tableName_R, _setting, model, avail_mem, lm);
            }
            
            delete [] model;
        }
        else if(option == option8)
        {
            // option 8: Materialize BCD
            string tableName = argv[2];
            int block_size = atoi(argv[3]);
            double step_size = atof(argv[4]);
            double error_tolerance = atof(argv[5]);
            int iter_num = atoi(argv[6]);
            const char *lm = argv[7];
            double avail_mem = atof(argv[8]);
            double *model;
            
            // Setting
            setting _setting;
            _setting.step_size = step_size;
            _setting.error = error_tolerance;
            _setting.iter_num = iter_num;
            
            t.materializeBCD(tableName, _setting, model, block_size, avail_mem, lm);
            
            delete [] model;
        }
        else
        {
            dM.errorMessage("Invalid command: wrong number of arguments or invalid option");
            exit(1);
        }
    }
    else if(argc == 10 && argv[1] == option9)
    {
        // option 9: Factorize BCD
        string tableName_S = argv[2];
        string tableName_R = argv[3];
        int block_size = atoi(argv[4]);
        double step_size = atof(argv[5]);
        double error_tolerance = atof(argv[6]);
        int iter_num = atoi(argv[7]);
        const char *lm = argv[8];
        double avail_mem = atof(argv[9]);
        double *model;
        
        // Setting
        setting _setting;
        _setting.step_size = step_size;
        _setting.error = error_tolerance;
        _setting.iter_num = iter_num;
        
        t.factorizeBCD(tableName_S, tableName_R, _setting, model, block_size, avail_mem, lm);
        
        delete [] model;

    }
    else
    {
        dM.errorMessage("Invalid command: wrong number of arguments or invalid option");
        exit(1);
    }
  
    return 0;
}

