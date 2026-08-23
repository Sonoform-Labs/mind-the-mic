$ErrorActionPreference = "Continue"
$dest = "data/diffrir"
$log = "$dest\_fetch2.log"
$rooms = @("dampenedBase", "hallwayBase", "complexBase")

function Log($m) {
  Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m) -Encoding utf8
}

Log "START $($rooms.Count) rooms"
foreach ($r in $rooms) {
  if (Test-Path "$dest\$r\RIRs.npy") { Log "SKIP $r"; continue }
  $zip = "$dest\$r.zip"
  Log "GET $r"
  & curl.exe -L -f -S -C - -o $zip "https://zenodo.org/records/11195833/files/$r.zip"
  Log "curl exit $LASTEXITCODE"
  if (Test-Path $zip) {
    try {
      Add-Type -AssemblyName System.IO.Compression.FileSystem
      [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $dest)
      Remove-Item $zip -Force
      Log "DONE $r"
    } catch { Log "extract FAILED $r : $($_.Exception.Message)" }
  }
  Log "free F: $([math]::Round((Get-PSDrive F).Free/1GB,1))GB"
}
Log "FINISH"
