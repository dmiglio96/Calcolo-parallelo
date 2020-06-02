import subprocess 
N = [30000]
M = 784
P = [20000]
LABELS = 10
K = 5

subprocess.call("rm resultsKNN_serial.out", shell= True)
lines = ["#ifndef INPUT\n", "#define INPUT\n", "#include <stdlib.h>\n", "#include <stdio.h>\n", "", "", "","", "#define LABELS 10\n", "typedef enum {true, false} bool;\n", "#endif\n"]
for i in range(len(N)):
	lines[4] = "#define M {}\n".format(M)
	lines[5] = "#define N {}\n".format(N[i])
	lines[6] = "#define P {}\n".format(P[i])
	lines[7] = "#define K {}\n".format(K)
	with open("input.h", "w") as f:
		for x in lines:
			f.write(x)
	print("Test con ", N[i])
	#for x in range(10):
		#subprocess.check_output(["make", "clean"])
	subprocess.check_output(["make"])
	subprocess.call(["./knn.x train2 test2"], shell = True)
