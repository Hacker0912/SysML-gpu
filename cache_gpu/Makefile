all: DB

DB: Learning.o DataManagement.o techniquescuda.o
	nvcc -std=c++11  -O3  -gencode arch=compute_61,code=sm_61 Learning.o DataManagement.o techniquescuda.o -o DB

Learning.o: Learning.cpp DataManagement.h techniques.h
	nvcc -std=c++11 -c -g -G  -O3  -gencode arch=compute_61,code=sm_61 Learning.cpp

DataManagement.o: DataManagement.cpp DataManagement.h
	nvcc -std=c++11 -c -g -G  -O3  -gencode arch=compute_61,code=sm_61 DataManagement.cpp

techniquescuda.o: techniquescuda.cu techniques.h DataManagement.h linear_models.h gradientkl.cu
	nvcc -std=c++11 -c -g -G  -O3  -gencode arch=compute_61,code=sm_61 techniquescuda.cu gradientkl.cu

clean:
	rm -f *.o DB *~ #*
