knn_openMP_openMP.x: knn_openMP.o knn_functions.o knn_openMP.o utility.o check.o 
	gcc -std=c99 -fopenmp -o knn_openMP.x knn_functions.o knn_openMP.o check.o utility.o -lm

knn_openMP.o: knn_openMP.c input.h 
	gcc -std=c99 -fopenmp -c knn_openMP.c -o knn_openMP.o -lm

knn_functions.o: knn_functions.c knn_functions.h input.h 
	gcc -std=c99 -fopenmp -c knn_functions.c -o knn_functions.o -lm

check.o: check.c check.h input.h 
	gcc -std=c99 -fopenmp -c check.c -o check.o -lm

utility.o: utility.c utility.h input.h 
	gcc -std=c99 -fopenmp -c utility.c -o utility.o -lm

.PHONY: clean

clean:
	rm -f *.o knn_openMP.x
