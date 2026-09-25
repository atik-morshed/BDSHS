# setup_and_run_gpu.ps1
# Helper script to activate bdshs-gpu conda env, ensure dependencies, and run bdshs_attribution.py using the GPU.
# Run this from the project directory (E:\BD SHS) in PowerShell:
#   .\setup_and_run_gpu.ps1

# Try to activate the conda environment (works if conda is initialized in this shell)
try {
    conda activate bdshs-gpu
} catch {
    Write-Host "conda activate not available in this shell; attempting to source activate script..."
    & "C:\\Users\\user7\\miniconda3\\Scripts\\activate" bdshs-gpu
}

# Show environment & GPU info
Write-Host "== Environment & GPU check =="
python -c "import sys; print('python', sys.executable)"
python -c "import torch; print('torch', getattr(torch,'__version__',None)); print('cuda_available', torch.cuda.is_available()); print('cuda_version', getattr(torch.version, 'cuda', None)); print('device_count', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no device')"

# Ensure core Python packages via conda (numpy, pandas)
Write-Host "== Installing/ensuring numpy,pandas via conda =="
conda install -n bdshs-gpu -y numpy pandas -c conda-forge

# Ensure transformers/captum/shap via pip inside the env
Write-Host "== Upgrading pip and installing transformers, captum, shap via pip =="
& "C:\\Users\\user7\\miniconda3\\envs\\bdshs-gpu\\python.exe" -m pip install --upgrade pip
& "C:\\Users\\user7\\miniconda3\\envs\\bdshs-gpu\\python.exe" -m pip install --upgrade transformers captum shap

# Run the attribution script with user-site disabled to avoid conflicts
Write-Host "== Running bdshs_attribution.py using bdshs-gpu env (GPU-enabled) =="
$env:PYTHONNOUSERSITE = "1"
& "C:\\Users\\user7\\miniconda3\\envs\\bdshs-gpu\\python.exe" bdshs_attribution.py

# If you prefer to run the script in the background, uncomment the following line:
# Start-Process -FilePath "C:\\Users\\user7\\miniconda3\\envs\\bdshs-gpu\\python.exe" -ArgumentList "bdshs_attribution.py" -NoNewWindow

Write-Host "Script finished (or exited with error). Check outputs in .\outputs\attribution and the log file attribution_log.txt"
