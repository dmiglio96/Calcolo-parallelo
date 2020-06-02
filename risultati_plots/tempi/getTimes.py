from statistics import *

#test seriale
with open("resultsKNN_serial.out", "r") as f:
	fl = f.readlines()
	time = []
	count = 0
	for x in fl:
		if len(x.strip()) != 0:
			l = []
			for t in x.split():			#ogni esperimento
				try:
					l.append(float(t))	
				except ValueError:
					pass
			
			time.append(l[4])
			
	
	print("tempi seriale", time)
	print("\n")

f.close()

times = {}	
unit = [2, 4, 8, 16, 32]
lenTest = 6
for i in unit:
	times[i] = []


#test openMP
with open("resultsKNN_openMP.out", "r") as f:
	fl = f.readlines()

	time = []
	c = 0
	for x in fl:
		if len(x.strip()) != 0:
			l = []
			for t in x.split():			#ogni esperimento
				try:
					l.append(float(t))	
				except ValueError:
					pass
			#print(l)
			time.append(l[5])
			if (len(time) == lenTest):
				
				times[unit[c]] = time
				c = c+1
				time = []

				
			
	
	print("tempi OPENMP", times)
	print("\n")

f.close()


#test mpi
with open("resultsKNN_mpi.out", "r") as f:
	fl = f.readlines()

	time = []
	c = 0
	for x in fl:
		if len(x.strip()) != 0:
			l = []
			for t in x.split():			#ogni esperimento
				try:
					l.append(float(t))	
				except ValueError:
					pass
			#print(l)
			time.append(l[6])
			if (len(time) == lenTest):
				
				times[unit[c]] = time
				c = c+1
				time = []

				
			
	
	print("tempi MPI", times)
	print("\n")
f.close()


#test mpiv2
with open("resultsKNN_mpiv2.out", "r") as f:
	fl = f.readlines()

	time = []
	c = 0
	for x in fl:
		if len(x.strip()) != 0:
			l = []
			for t in x.split():			#ogni esperimento
				try:
					l.append(float(t))	
				except ValueError:
					pass
			#print(l)
			time.append(l[6])
			if (len(time) == lenTest):
				
				times[unit[c]] = time
				c = c+1
				time = []

				
			
	
	print("tempi MPI versione2", times)
	print("\n")

f.close()

#test cuda
times = {}
block = [8, 16, 32]
for i in block:
	times[i] = []

with open("resultsKNN_cudav2.out", "r") as f:
	fl = f.readlines()

	time = []
	c = 0
	for x in fl:
		if len(x.strip()) != 0:
			l = []
			for t in x.split():			#ogni esperimento
				try:
					l.append(float(t))	
				except ValueError:
					pass
			#print(l)
			time.append(l[5])
			if (len(time) == lenTest):
				
				times[block[c]] = time
				c = c+1
				time = []

				
			
	
	print("tempi CUDA", times)
	print("\n")

f.close()

print("\n--------- Secondo dataset  -------------\n")

with open("resultsKNN_serialAttr.out", "r") as f:
	fl = f.readlines()
	time = []
	count = 0
	for x in fl:
		if len(x.strip()) != 0:
			l = []
			for t in x.split():			#ogni esperimento
				try:
					l.append(float(t))	
				except ValueError:
					pass
			
			time.append(l[4])
			
	
	print("tempi seriale", time)
	print("\n")

f.close()

times = {}	
unit = [2, 4, 8, 16, 32]
lenTest = 1
for i in unit:
	times[i] = []


with open("resultsKNN_openMPAttr.out", "r") as f:
	fl = f.readlines()

	time = []
	c = 0
	for x in fl:
		if len(x.strip()) != 0:
			l = []
			for t in x.split():			#ogni esperimento
				try:
					l.append(float(t))	
				except ValueError:
					pass
			#print(l)
			time.append(l[5])
			if (len(time) == lenTest):
				
				times[unit[c]] = time
				c = c+1
				time = []

				
			
	
	print("tempi OPENMP", times)
	print("\n")	
f.close()


with open("resultsKNN_mpiAttr.out", "r") as f:
	fl = f.readlines()

	time = []
	c = 0
	for x in fl:
		if len(x.strip()) != 0:
			l = []
			for t in x.split():			#ogni esperimento
				try:
					l.append(float(t))	
				except ValueError:
					pass
			#print(l)
			time.append(l[6])
			if (len(time) == lenTest):
				
				times[unit[c]] = time
				c = c+1
				time = []

				
			
	
	print("tempi MPI", times)
	print("\n")
f.close()

with open("resultsKNN_mpiv2Attr.out", "r") as f:
	fl = f.readlines()

	time = []
	c = 0
	for x in fl:
		if len(x.strip()) != 0:
			l = []
			for t in x.split():			#ogni esperimento
				try:
					l.append(float(t))	
				except ValueError:
					pass
			#print(l)
			time.append(l[6])
			if (len(time) == lenTest):
				
				times[unit[c]] = time
				c = c+1
				time = []

				
			
	
	print("tempi MPI versione2", times)
	print("\n")

f.close()

times = {}
block = [8, 16, 32]
for i in block:
	times[i] = []

with open("resultsKNN_cudav2Attr.out", "r") as f:
	fl = f.readlines()

	time = []
	c = 0
	for x in fl:
		if len(x.strip()) != 0:
			l = []
			for t in x.split():			#ogni esperimento
				try:
					l.append(float(t))	
				except ValueError:
					pass
			#print(l)
			time.append(l[5])
			if (len(time) == lenTest):
				
				times[block[c]] = time
				c = c+1
				time = []

				
			
	
	print("tempi CUDA", times)
	print("\n")




