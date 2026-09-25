# Train BD-SHS on this machine's RTX A6000 using the E: venv (does not use C: conda envs).
Set-Location $PSScriptRoot
$env:PYTHONNOUSERSITE = '1'
$env:HF_HOME = Join-Path $PSScriptRoot '.cache\huggingface'
$env:HUGGINGFACE_HUB_CACHE = Join-Path $PSScriptRoot '.cache\huggingface\hub'
$env:TRANSFORMERS_CACHE = Join-Path $PSScriptRoot '.cache\huggingface\transformers'
$env:TMP = Join-Path $PSScriptRoot '.cache\tmp'
$env:TEMP = Join-Path $PSScriptRoot '.cache\tmp'

$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Error "Missing $py - recreate the E: environment first."
    exit 1
}

$check = @'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)
'@
$check | & $py -
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $py -u (Join-Path $PSScriptRoot 'bdshs_pipeline.py')
