$ErrorActionPreference = 'Stop'

$envDir = 'Y:\tools\nexalang-spirv'
$bin1 = Join-Path $envDir 'Library\bin'
$bin2 = Join-Path $envDir 'Scripts'

if (!(Test-Path $bin1)) {
  throw "SPIR-V env not found at '$envDir'. Create it first (see README / tools/check_spirv_toolchain.py)."
}

$env:Path = "$bin1;$bin2;$env:Path"

Write-Host "Activated NexaLang SPIR-V toolchain for this session."
Write-Host "  llvm-as:   " (Get-Command llvm-as -ErrorAction SilentlyContinue).Source
Write-Host "  llc:       " (Get-Command llc -ErrorAction SilentlyContinue).Source
Write-Host "  llvm-spirv:" (Get-Command llvm-spirv -ErrorAction SilentlyContinue).Source
Write-Host "  spirv-val: " (Get-Command spirv-val -ErrorAction SilentlyContinue).Source


