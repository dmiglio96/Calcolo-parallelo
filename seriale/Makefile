knn.x: knn.o knn_functions.o knn.o utility.o 
	gcc -std=c99 -o knn.x knn_functions.o knn.o utility.o -lm

knn.o: knn.c input.h 
	gcc -std=c99 -c knn.c -o knn.o -lm

knn_functions.o: knn_functions.c knn_functions.h input.h 
	gcc -std=c99 -c knn_functions.c -o knn_functions.o -lm

utility.o: utility.c utility.h input.h 
	gcc -std=c99 -c utility.c -o utility.o -lm

.PHONY: clean

clean:
	rm -f *.o knn.x
