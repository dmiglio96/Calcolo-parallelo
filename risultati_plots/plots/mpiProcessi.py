from pylab import * 
from numpy import *

if __name__ == '__main__':
    test = [480 ,800, 8000, 16000, 32000, 64000]
    NP = [2, 4, 8, 16, 32]
    mpi = {2: [0.265816, 0.299116, 1.920187, 4.8794, 16.135056, 64.224754], 4: [0.280108, 0.296003, 1.426094, 3.049108, 8.78381, 33.711452], 8: [0.308802, 0.334361, 1.067778, 2.169318, 5.367494, 17.170334], 16: [0.411926, 0.43894, 0.945612, 1.663832, 3.649362, 10.054904], 32: [0.465732, 0.965883, 1.242874, 1.848685, 3.773326, 10.197095]}

    colors = "cyan-darkturquoise-deepskyblue-dodgerblue-royalblue-blue".split('-')
    figure(figsize=(8,6))
    distanze = [[0.25, 15.75, 31.25, 46.75, 62.25, 77.75],
[2.25, 17.75, 33.25, 48.75, 64.25, 79.75],
[4.25, 19.75, 35.25, 50.75, 66.25, 81.75],
[6.25, 21.75, 37.25, 52.75, 68.25, 83.75],
[8.25, 23.75, 39.25, 54.75, 70.25, 85.75],
[10.25, 25.75, 41.25, 56.75, 72.25, 87.75]]

    #bar(distanze[0], mpi[1] , width=1.5, color=colors[0], ec="black", label="1 Processo")
    bar(distanze[0], mpi[2] , width=1.5,  color=colors[1], ec="black", label="2 Processi")
    bar(distanze[1], mpi[4] , width=1.5, color=colors[2], ec="black", label="4 Processi")
    bar(distanze[2], mpi[8] , width=1.5, color=colors[3], ec="black",  label="8 Processi")
    bar(distanze[3], mpi[16] , width=1.5,  color=colors[4], ec="black", label="16 Processi")
    bar(distanze[4], mpi[32] , width=1.5, color=colors[5], ec="black", label="32 Processi")
    
    xticks([5, 20.5, 36, 51.5, 67, 82.5], test)
    title("Confronto tempo al variare del numero di processi", fontsize=16)
    yscale('log')
    xlabel("Dimensione dataset", fontsize=14)
    ylabel("Tempo richiesto", fontsize=14)

    legend()
    savefig("processiMPI.pdf")
    close()

    #--------------------------------------------------#
    #secondo dataset
    test = ["30000 x 784"]
    NP = [2, 4, 8, 16, 32]
    mpi = {2: [784.120361], 4: [416.832733], 8: [219.725204], 16: [118.374481], 32: [119.797905]}

    colors = "cyan-darkturquoise-deepskyblue-dodgerblue-royalblue-blue".split('-')
    figure(figsize=(8,6))
    distanze = [[0.25], [2.25], [4.25], [6.25], [8.25], [10.25]]

    #bar(distanze[0], mpi[1] , width=1.5, color=colors[0], ec="black", label="1 Processo")
    bar(distanze[0], mpi[2] , width=1.5,  color=colors[1], ec="black", label="2 Processi")
    bar(distanze[1], mpi[4] , width=1.5, color=colors[2], ec="black", label="4 Processi")
    bar(distanze[2], mpi[8] , width=1.5, color=colors[3], ec="black",  label="8 Processi")
    bar(distanze[3], mpi[16] , width=1.5,  color=colors[4], ec="black", label="16 Processi")
    bar(distanze[4], mpi[32] , width=1.5, color=colors[5], ec="black", label="32 Processi")
    
    xticks([5],test)
    title("Confronto tempo al variare del numero di processi dataset 30000 x 784", fontsize=12)
    yscale('log')
    #xlabel("Dimensione dataset", fontsize=14)
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    ylabel("Tempo richiesto", fontsize=14)

    legend()
    savefig("processiMPI_attr.pdf")
    close()

#------------------------------------------------------------------------------------#
    #versione 2
    test = [480 ,800, 8000, 16000, 32000, 64000]
    NP = [2, 4, 8, 16, 32]

    colors = "cyan-darkturquoise-deepskyblue-dodgerblue-royalblue-blue".split('-')
    figure(figsize=(8,6))
    distanze = [[0.25, 15.75, 31.25, 46.75, 62.25, 77.75],
[2.25, 17.75, 33.25, 48.75, 64.25, 79.75],
[4.25, 19.75, 35.25, 50.75, 66.25, 81.75],
[6.25, 21.75, 37.25, 52.75, 68.25, 83.75],
[8.25, 23.75, 39.25, 54.75, 70.25, 85.75],
[10.25, 25.75, 41.25, 56.75, 72.25, 87.75]]
    mpiv2 ={2: [0.265473, 0.298741, 1.947798, 5.039396, 16.97044, 63.899754], 4: [0.266858, 0.301406, 1.440811, 3.256303, 9.169259, 32.749729], 8: [0.309422, 0.352757, 1.095037, 2.178585, 5.419026, 17.44927], 16: [0.376384, 0.462459, 0.937736, 1.810122, 3.494534, 9.498455], 32: [0.449857, 0.613673, 1.240905, 1.981682, 3.730247, 9.745857]}
bar(distanze[0], mpiv2[2] , width=1.5,  color=colors[1], ec="black", label="2 Processi")
bar(distanze[1], mpiv2[4] , width=1.5, color=colors[2], ec="black", label="4 Processi")
bar(distanze[2], mpiv2[8] , width=1.5, color=colors[3], ec="black",  label="8 Processi")
bar(distanze[3], mpiv2[16] , width=1.5,  color=colors[4], ec="black", label="16 Processi")
bar(distanze[4], mpiv2[32] , width=1.5, color=colors[5], ec="black", label="32 Processi")
	    
xticks([5, 20.5, 36, 51.5, 67, 82.5], test)
title("Confronto tempo al variare del numero di processi", fontsize=16)
yscale('log')
xlabel("Dimensione dataset", fontsize=14)
ylabel("Tempo richiesto", fontsize=14)

legend()
savefig("processiMPIv2.pdf")
close()




	#--------------------------------------------------#
	#secondo dataset
test = ["30000 x 784"]
NP = [2, 4, 8, 16, 32]

colors = "cyan-darkturquoise-deepskyblue-dodgerblue-royalblue-blue".split('-')
figure(figsize=(8,6))
distanze = [[0.25], [2.25], [4.25], [6.25], [8.25], [10.25]]
mpiv2 = {2: [811.649231], 4: [422.972229], 8: [218.037796], 16: [120.108658], 32: [116.828667]}

bar(distanze[0], mpiv2[2] , width=1.5,  color=colors[1], ec="black", label="2 Processi")
bar(distanze[1], mpiv2[4] , width=1.5, color=colors[2], ec="black", label="4 Processi")
bar(distanze[2], mpiv2[8] , width=1.5, color=colors[3], ec="black",  label="8 Processi")
bar(distanze[3], mpiv2[16] , width=1.5,  color=colors[4], ec="black", label="16 Processi")
bar(distanze[4], mpiv2[32] , width=1.5, color=colors[5], ec="black", label="32 Processi")
	    
xticks([5], test)
title("Confronto tempo al variare del numero di processi dataset 30000 x 784", fontsize=12)
yscale('log')
#xlabel("Dimensione dataset", fontsize=14)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ylabel("Tempo richiesto", fontsize=14)

legend()
savefig("processiMPIv2_attr.pdf")
close()

