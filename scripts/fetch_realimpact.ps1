$ErrorActionPreference = "Continue"
$pkg = Split-Path -Parent $PSScriptRoot
$dest = if ($env:MTM_REALIMPACT_DIR) { $env:MTM_REALIMPACT_DIR } else { Join-Path $pkg "data\realimpact" }
$base = "http://downloads.cs.stanford.edu/viscam/RealImpact"
$objects = @("3_CeramicKoiBowl","10_bowl","17_IronSkillet","19_Pan","27_WoodPlate","31_WoodSlab","32_WoodChalice","34_WoodMug","43_IronMortar","48_PlasticBowl","49_PlasticBowl","61_PlanterCeramic","64_CeramicMug","67_IronPlate","73_PlasticBin","78_CeramicCup","83_WoodVase","91_MetalSpoon","94_GlassGoblet","97_PlasticScoop")
if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Force $dest | Out-Null }
Add-Type -AssemblyName System.IO.Compression.FileSystem
foreach ($o in $objects) {
  if (Test-Path (Join-Path $dest "$o\preprocessed\deconvolved_0db.npy")) { Write-Output "skip $o"; continue }
  $zip = Join-Path $dest "$o.zip"
  Write-Output "get $o"
  & curl.exe -L -f -S -C - -o $zip "$base/$o.zip"
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $zip)) { Write-Output "failed $o"; continue }
  try {
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $dest)
    Remove-Item $zip -Force
  } catch {
    Write-Output "extract failed $o : $($_.Exception.Message)"
  }
}
$n = ($objects | Where-Object { Test-Path (Join-Path $dest "$_\preprocessed\deconvolved_0db.npy") }).Count
Write-Output "$n of 20 objects ready in $dest"
