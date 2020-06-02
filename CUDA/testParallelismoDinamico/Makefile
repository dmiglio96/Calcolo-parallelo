knn_cuda.x: knn_cuda.o knn_functions.o utility.o check.o 
	nvcc -Xptxas= --use_fast_math -gencode arch=compute_35,code=sm_35 -rdc=true -lcudadevrt -o knn_cuda.x knn_cuda.o knn_functions.o utility.o check.o 

knn_cuda.o: knn_cuda.cu input.h 
	nvcc -Xptxas= --use_fast_math -gencode arch=compute_35,code=sm_35 -rdc=true -lcudadevrt -c knn_cuda.cu -o knn_cuda.o

check.o: check.cu check.h input.h 
	nvcc -Xptxas= --use_fast_math -gencode arch=compute_35,code=sm_35 -rdc=true -lcudadevrt -c  check.cu -o check.o

knn_functions.o: knn_functions.cu knn_functions.h input.h 
	nvcc -Xptxas= --use_fast_math -gencode arch=compute_35,code=sm_35 -rdc=true -lcudadevrt -c knn_functions.cu -o knn_functions.o

utility.o: utility.cu utility.h input.h 
	nvcc -Xptxas= --use_fast_math -gencode arch=compute_35,code=sm_35 -rdc=true -lcudadevrt -c utility.cu -o utility.o 

.PHONY: clean

clean:
	rm -f *.o knn_cuda.x
