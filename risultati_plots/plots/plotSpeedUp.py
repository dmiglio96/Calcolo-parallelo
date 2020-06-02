from matplotlib import pylab 
from pylab import *
#speedup
if __name__ == '__main__':
    test = [480 ,800, 8000, 16000, 32000, 64000]

    seriale = [0.01, 0.06, 2.25, 7.93, 29.74, 120.830002]

    cuda = [0.011303, 0.021059, 0.202351, 0.432082, 0.913778, 2.230188]
    omp = [0.005187, 0.024208, 0.711001, 1.515463, 3.307396, 9.554554]
    mpi = [0.376384, 0.462459, 0.937736, 1.810122, 3.494534, 9.498455]
    colors = "royalblue-blueviolet-darkblue".split('-')
    
    #calcolo speedup
    for i in range(len(seriale)):
        cuda[i] = seriale[i]/cuda[i]
        omp[i] = seriale[i]/omp[i]
        mpi[i] = seriale[i]/mpi[i]

    figure(figsize=(8,6))
    bar([0.25, 5.25, 10.25, 15.25, 20.25, 25.25], omp , width=1.5, color=colors[0], ec="black", label="OpenMP")
    bar([1.75,6.75,11.75,16.75, 21.75, 26.75], mpi , width=1.5,  color=colors[1], ec="black", label="MPI")
    bar([2.5, 7.5, 12.5, 17.5, 22.5, 27.5], cuda , width=1.5, color=colors[2], ec="black", align="edge", label="CUDA")
    
    title("Confronto Speed-up - dimensione dataset", fontsize=16)
    xlabel("Dimensione dataset", fontsize=14)
    ylabel("Speed-up", fontsize=14)
    legend(loc="upper left")
    xticks([1.75, 6.75, 11.75, 16.75, 21.75, 26.75], test)
    #yscale("log")
    tight_layout()
    savefig("speedUP.pdf")
    close()


    #----------------------------
    test = ["30000 x 784"]
    #dataset 2
    seriale = [1405.27002]
	
    #tempi 
    cuda = [23.299089]
    omp = [115.306534]
    mpi = [118.374481]
    colors = "royalblue-blueviolet-darkblue".split('-')
    
    #calcolo speedup
    for i in range(len(seriale)):
        cuda[i] = seriale[i]/cuda[i]
        omp[i] = seriale[i]/omp[i]
        mpi[i] = seriale[i]/mpi[i]

    figure(figsize=(8,6))
    bar([0.25], omp , width=1, color=colors[0], ec="black", label="OpenMP")
    bar([1.75], mpi , width=1,  color=colors[1], ec="black", label="MPI")
    bar([2.75], cuda , width=1, color=colors[2], ec="black", align="edge", label="CUDA")
    
    title("Confronto Speed-up dataset 30000 x 784", fontsize=16)
    #xlabel("Dimensione dataset", fontsize=14)
    ylabel("Speed-up", fontsize=14)
    legend(loc="upper left")
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    #xticks([1.75], test)
    #yscale("log")
    tight_layout()
    savefig("speedUP_attr.pdf")
    close()
