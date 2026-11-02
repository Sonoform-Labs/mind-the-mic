$ErrorActionPreference = "Continue"
$pkg = Split-Path -Parent $PSScriptRoot
$dest = if ($env:MTM_DIFFRIR_DIR) { $env:MTM_DIFFRIR_DIR } else { Join-Path $pkg "data\diffrir" }
$base = "https://zenodo.org/records/11195833/files"
$rooms = if ($env:MTM_ROOMS) { $env:MTM_ROOMS -split "\s+" } else { @("classroomBase", "dampenedBase", "hallwayBase", "complexBase") }
if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Force $dest | Out-Null }
Add-Type -AssemblyName System.IO.Compression.FileSystem
foreach ($r in $rooms) {
  if (Test-Path (Join-Path $dest "$r\RIRs.npy")) { Write-Output "skip $r"; continue }
  $zip = Join-Path $dest "$r.zip"
  Write-Output "get $r"
  & curl.exe -L -f -S -C - -o $zip "$base/$r.zip"
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $zip)) { Write-Output "failed $r"; continue }
  try {
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $dest)
    Remove-Item $zip -Force
  } catch {
    Write-Output "extract failed $r : $($_.Exception.Message)"
  }
}
foreach ($r in $rooms) {
  if (Test-Path (Join-Path $dest "$r\RIRs.npy")) { Write-Output "ready $r" } else { Write-Output "missing $r" }
}
