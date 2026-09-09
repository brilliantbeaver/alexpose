<#
Create an isolated CUDA kernel using the existing environment's package versions.
Run from any directory. Existing notebook kernels are left running.
Official wheel source: https://pytorch.org/get-started/previous-versions/
#>
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$basePython = Join-Path $taskRoot '.venv/Scripts/python.exe'
$cudaEnvironment = Join-Path $taskRoot '.venv-cuda'
$cudaPython = Join-Path $cudaEnvironment 'Scripts/python.exe'
$taskUv = (Get-Command uv -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath $basePython)) {
    throw 'Prepare the project .venv first; it supplies the dependency versions to preserve.'
}
if (-not (Test-Path -LiteralPath $cudaPython)) {
    & $taskUv venv $cudaEnvironment --python $basePython
    if ($LASTEXITCODE -ne 0) { throw 'CUDA virtual environment creation failed.' }
}
$taskRequirements = Join-Path ([System.IO.Path]::GetTempPath()) ('gavd-cuda-' + [guid]::NewGuid() + '.txt')
try {
    $taskPackages = & $taskUv pip freeze --python $basePython
    if ($LASTEXITCODE -ne 0) { throw 'Could not read existing dependency versions.' }
    $taskPackages | Where-Object { $_ -notmatch '^(torch|torchvision|torchaudio)==' } |
        Set-Content -LiteralPath $taskRequirements -Encoding utf8
    & $taskUv pip install --python $cudaPython 'torch==2.13.0' --index-url 'https://download.pytorch.org/whl/cu130'
    if ($LASTEXITCODE -ne 0) { throw 'Official CUDA PyTorch installation failed.' }
    & $taskUv pip install --python $cudaPython -r $taskRequirements
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $cudaPython -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name()); print((torch.ones(32,32,device='cuda') @ torch.ones(32,32,device='cuda')).mean().item())"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA computation check failed; inspect the driver and wheel before training.' }
    & $cudaPython -m ipykernel install --user --name gavd5-cuda --display-name 'GAVD5 CUDA (PyTorch 2.13)'
    if ($LASTEXITCODE -ne 0) { throw 'Kernel registration failed.' }
    Write-Host 'Select GAVD5 CUDA (PyTorch 2.13) in Notebooks 17 and 18, then run from the top.'
} finally {
    Remove-Item -LiteralPath $taskRequirements -ErrorAction SilentlyContinue
}
