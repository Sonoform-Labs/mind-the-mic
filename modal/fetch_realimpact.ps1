$ErrorActionPreference = "Continue"
$dest = "data/realimpact"
$log  = "data/realimpact\_fetch_log.txt"
$base = "http://downloads.cs.stanford.edu/viscam/RealImpact"

$objects = @(
  "3_CeramicKoiBowl","64_CeramicMug","78_CeramicCup","61_PlanterCeramic",
  "27_WoodPlate","32_WoodChalice","34_WoodMug","83_WoodVase",
  "19_Pan","67_IronPlate","43_IronMortar","91_MetalSpoon",
  "49_PlasticBowl","73_PlasticBin","97_PlasticScoop","94_GlassGoblet"
)

function Log($m) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Force $dest | Out-Null }
Log "START pid=$PID  $($objects.Count) objects -> $dest"

$done = 0; $skip = 0; $fail = 0
foreach ($o in $objects) {
  $odir = Join-Path $dest $o
  if (Test-Path (Join-Path $odir "preprocessed\deconvolved_0db.npy")) {
    Log "SKIP $o (already present)"; $skip++; continue
  }
  $zip = Join-Path $dest "$o.zip"
  $url = "$base/$o.zip"
  Log "GET  $o"
  & curl.exe -L -f -s -S -C - -o $zip $url 2>&1 | ForEach-Object { Log "     curl: $_" }
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $zip)) {
    Log "FAIL $o (curl exit $LASTEXITCODE)"; $fail++
    if (Test-Path $zip) { Remove-Item $zip -Force -ErrorAction SilentlyContinue }
    continue
  }
  $mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
  Log "OK   $o downloaded ${mb}MB, extracting"
  try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $dest)
    Remove-Item $zip -Force
    $done++
    Log "DONE $o extracted"
  } catch {
    Log "FAIL $o extract: $($_.Exception.Message)"; $fail++
  }
  $free = [math]::Round((Get-PSDrive F).Free / 1GB, 1)
  Log "     free on F: ${free}GB"
}
Log "FINISH downloaded=$done skipped=$skip failed=$fail"
$n = (Get-ChildItem $dest -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'preprocessed\deconvolved_0db.npy') }).Count
Log "objects now usable: $n"
