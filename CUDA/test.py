import subprocess
N = [480 ,800, 8000, 16000, 32000, 64000] 
M = 30
P = [20, 200, 2000, 4000, 8000, 16000]
K = 5
LABELS = 10
BLOCK_SIZE = [8, 16, 32]

subprocess.call("rm resultsKNN_cudav2.out", shell= True)
lines = ["#ifndef INPUT\n", "#define INPUT\n", "#include <stdlib.h>\n", "#include <stdio.h>\n", "", "", "","", "", "#define LABELS 10\n", "#endif\n"]
for j in range(len(BLOCK_SIZE)):
	for i in range(len(N)):
		lines[4] = "#define N {}\n".format(N[i])
		lines[5] = "#define P {}\n".format(P[i])
		lines[6] = "#define K {}\n".format(K)
		lines[7] = "#define BLOCK_SIZE {}\n".format(BLOCK_SIZE[j])
		lines[8] = "#define M {}\n".format(M)
		with open("input.h", "w") as f:
			for x in lines:
				f.write(x)
		print "TEST WITH = ", N[i],
		print "BLOCK_SIZE = ", BLOCK_SIZE[j]
		for x in range(1):
			#subprocess.check_output(["make", "clean"])
			subprocess.check_output(["make"])
			subprocess.call("./knn_cuda.x train test", shell= True)
	
