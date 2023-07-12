import subprocess

def assert_gpu_runtime():
    error_message = '\n\n⚠️ ERROR: GPU is not detected! ⚠️\n\nPlease change the runtime type via:\n\tSelect "Runtime" > "Change runtime type" > "Hardware accelerator" > "GPU"\nand re-run the notebook.'
    sp_call = subprocess.call('nvidia-smi', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # run `nvidia-smi` to see if we can see the GPU, if not raise AssertionError
    assert  sp_call == 0, f'{error_message}'

