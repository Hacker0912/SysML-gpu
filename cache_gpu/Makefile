all: DB

DB: Learning.o DataManagement.o techniquescuda.o
	nvcc -std=c++11  -O3  -gencode arch=compute_61,code=sm_61 Learning.o DataManagement.o techniquescuda.o -o DB -Xcompiler -fopenmp -lpthread

Learning.o: Learning.cpp DataManagement.h techniques.h
	nvcc -std=c++11 -c -g -G  -O3  -gencode arch=compute_61,code=sm_61 Learning.cpp -Xcompiler -fopenmp -lpthread

DataManagement.o: DataManagement.cpp DataManagement.h
	nvcc -std=c++11 -c -g -G  -O3  -gencode arch=compute_61,code=sm_61 DataManagement.cpp -Xcompiler -fopenmp -lpthread

techniquescuda.o: techniquescuda.cu techniques.h DataManagement.h linear_models.h gradientkl.cu
	nvcc -std=c++11 -c -g -G  -O3  -gencode arch=compute_61,code=sm_61 techniquescuda.cu gradientkl.cu -Xcompiler  -fopenmp -lpthread

clean:
	rm -f *.o DB *~ #*
