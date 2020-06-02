knn_mpi.x: knn_mpi.o knn_functions.o utility.o check.o 
	mpicc -std=c99 -o knn_mpi.x knn_functions.o knn_mpi.o utility.o check.o -lm

knn_mpi.o: knn_mpi.c input.h 
	mpicc -std=c99 -c knn_mpi.c -o knn_mpi.o -lm

check.o: check.c check.h input.h 
	mpicc -std=c99 -c check.c -o check.o

knn_functions.o: knn_functions.c knn_functions.h input.h utility.o
	mpicc -std=c99 -c knn_functions.c -o knn_functions.o -lm

utility.o: utility.c utility.h input.h 
	mpicc -std=c99 -c utility.c -o utility.o -lm

.PHONY: clean

clean:
	rm -f *.o knn_mpi.x

