from pylab import * 
from numpy import *

if __name__ == '__main__':
    test = [480 ,800, 8000, 16000, 32000, 64000]

    serial = [0.01, 0.06, 2.25, 7.93, 29.74, 120.830002]

    cuda = [0.011303, 0.021059, 0.202351, 0.432082, 0.913778, 2.230188]
    omp = [0.005187, 0.024208, 0.711001, 1.515463, 3.307396, 9.554554]
    mpi = [0.376384, 0.462459, 0.937736, 1.810122, 3.494534, 9.498455]

    colors = "darkturquoise-royalblue-blueviolet-darkblue".split('-')

    figure(figsize=(8,6))
    plot(test, serial, color=colors[0], linestyle='-',label='Serial', marker='o')
    plot(test, omp, color=colors[1], linestyle='-', label= 'OmpenMP', marker='o')
    plot(test, mpi, color=colors[2], linestyle='-', label='MPI', marker='o')
    plot(test, cuda, color=colors[3], linestyle='-', label='CUDA', marker='o')
    
    title("Confronto tempo totale - dimensione dataset", fontsize=16)
    #yscale('log')
    xlabel("Dimensione dataset", fontsize=14)
    ylabel("Tempo richiesto", fontsize=14)
    legend()
    savefig("times.pdf")
    close()


    #------------------------------------
    #dataset 2
    test = ["30000 x 784"]
    #dataset 2
    seriale = [1405.27002]
	
    #tempi 
    cuda = [23.299089]
    omp = [115.306534]
    mpi = [118.374481]
    colors = "darkturquoise-royalblue-blueviolet-darkblue".split('-')
    

    figure(figsize=(8,6))
    bar([0.25], seriale , width=1.25, color=colors[0], ec="black", label="Seriale")
    bar([1.75], omp , width=1.25, color=colors[1], ec="black", label="OpenMP")
    bar([3.25], mpi , width=1.25,  color=colors[2], ec="black", label="MPI")
    bar([4.10], cuda , width=1.25, color=colors[3], ec="black", align="edge", label="CUDA")
    
    title("Confronto tempo totale dataset 30000 x 784", fontsize=16)
    #xlabel("Dimensione dataset", fontsize=14)
    ylabel("Tempo richiesto", fontsize=14)
    legend(loc="upper left")

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

    yscale("log")
    tight_layout()
    savefig("times_attr.pdf")
    close()
