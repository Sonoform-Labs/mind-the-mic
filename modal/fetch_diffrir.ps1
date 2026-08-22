$ErrorActionPreference = "Continue"
$dest = "data/diffrir"
$log = "$dest\_fetch.log"
$zip = "$dest\classroomBase.zip"
$url = "https://zenodo.org/records/11195833/files/classroomBase.zip"

function Log($m) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
  Add-Content -Path $log -Value $line -Encoding utf8
}

if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Force $dest | Out-Null }
Log "START pid=$PID classroomBase ~4.99GB"

& curl.exe -L -f -S -C - -o $zip $url
Log "curl exit $LASTEXITCODE"

if (Test-Path $zip) {
  $mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
  Log "downloaded ${mb}MB, extracting"
  try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $dest)
    Remove-Item $zip -Force
    Log "EXTRACTED ok"
  } catch {
    Log "extract FAILED: $($_.Exception.Message)"
  }
} else {
  Log "no zip produced"
}
$n = (Get-ChildItem $dest -Recurse -File -ErrorAction SilentlyContinue).Count
Log "DONE, $n files under $dest"
