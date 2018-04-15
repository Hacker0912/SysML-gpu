//
//  linear_models.h
//  Coordinate_descent
//
//  Created by Zhiwei Fan on 10/14/15.
//  Copyright © 2015 Zhiwei Fan. All rights reserved.
//

#ifndef _linear_models_
#define _linear_models_

#include <stdio.h>
#include <string.h>
#include <cmath>
#define MAX(x,y) (x > y ? x : y)

inline double Fe_lr(double a, double b)
{
    return log(1+pow(exp(1.00), -(a*b)));
}

inline double Fe_lsr(double a, double b)
{
    return pow(a-b, 2);
}

inline double Fe_lsvm(double a, double b)
{
    return MAX(0, 1 - a*b);
}

inline double G_lr(double a, double b)
{
    return -(a/(1+pow(exp(1.0),a*b)));
}

inline double G_lsr(double a, double b)
{
    return 2*(b-a);
}

inline double G_svm(double a, double b)
{
    return  a*b > 1 ? -1:0;
}

inline double C_lr(double a)
{
    return 1.00/(double)(1+pow(exp(1.00),-a));
}

double gradientCompute(double a, double b, const char* lm)
{
    if(strcmp("lr",lm) == 0)
    {
        return G_lr(a,b);
    }
    else if(strcmp("lsr",lm) == 0)
    {
        return G_lsr(a,b);
    }
    else if(strcmp("lsvm",lm) == 0)
    {
        return G_svm(a,b);
    }
    else
    {
        fprintf(stderr, "Non-valid option for linear models");
        exit(1);
    }
}

double lossCompute(double a, double b, const char* lm)
{
    if(strcmp("lr",lm) == 0)
    {
        return Fe_lr(a,b);
    }
    else if(strcmp("lsr",lm) == 0)
    {
        return Fe_lsr(a,b);
    }
    else if(strcmp("lsvm",lm) == 0)
    {
        return Fe_lsvm(a,b);
    }
    else
    {
        fprintf(stderr, "Non-valid option for linear models");
        exit(1);
    }
}

#endif /* _linear_models_ */
