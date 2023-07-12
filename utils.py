import subprocess

def assert_gpu_runtime():
    assert subprocess.call('nvidia-smi', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0, \
        '\n\nGPU is not detected! Please change the runtime type:' \
        '\nSelect Runtime > "Change runtime type" > Select "GPU" under "Hardware accelerator" > re-run the notebook.'