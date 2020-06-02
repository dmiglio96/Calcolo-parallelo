import subprocess 

subprocess.call("python3 CUDA_block.py", shell= True)
subprocess.call("python3 mpiProcessi.py", shell= True)
subprocess.call("python3 openMP_thread.py", shell= True)
subprocess.call("python3 plotSpeedUp.py", shell= True)
subprocess.call("python3 plotTime.py", shell= True)