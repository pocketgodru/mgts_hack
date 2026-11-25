[CmdletBinding()]
param(
    [ValidateSet('cpu', 'auto', '0')]
    [string]$Device = 'auto',
    [ValidateRange(1024, 65535)]
    [int]$Port = 8000,
    [string]$Python = (Join-Path $PSScriptRoot '.venv/Scripts/python.exe'),
    [switch]$UseProjectPackages
)

$ErrorActionPreference = 'Stop'
if ($UseProjectPackages) {
    $projectPackages = Join-Path $PSScriptRoot '.venv/Lib/site-packages'
    if (-not (Test-Path -LiteralPath $projectPackages)) {
        throw "Project packages not found: $projectPackages"
    }
    $env:PYTHONPATH = $projectPackages
}
$env:MGTS_DEVICE = $Device
& $Python -m uvicorn web_app:app --app-dir $PSScriptRoot --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
