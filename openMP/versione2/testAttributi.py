import subprocess 
N = [30000]
M = 784
P = [20000]
LABELS = 10
K = 5
LABELS = 10
NT = [2, 4, 8, 16, 32]


subprocess.call("rm resultsKNN_openMPv2.out", shell= True)
lines = ["#ifndef INPUT\n", "#define INPUT\n", "#include <stdlib.h>\n", "#include <stdio.h>\n", "", "", "","", "", "#define LABELS 10\n", "typedef enum {true, false} bool;\n", "#endif\n"]
for j in range(len(NT)):
	for i in range(len(N)):
		lines[4] = "#define M {}\n".format(M)
		lines[5] = "#define N {}\n".format(N[i])
		lines[6] = "#define P {}\n".format(P[i])
		lines[7] = "#define K {}\n".format(K)
		lines[8] = "#define NT {}\n".format(NT[j])
		with open("input.h", "w") as f:
			for x in lines:
				f.write(x)
		print("Test con ", N[i])
		print("Usando ", NT[j])
		#for x in range(10):
			#subprocess.check_output(["make", "clean"])
		subprocess.check_output(["make"])
		subprocess.call(["./knn_openMP.x train2 test2"], shell = True)
