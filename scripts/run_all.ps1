param(
    [switch]$DownloadData
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

Push-Location $ProjectRoot
try {
    if (-not (Test-Path '.venv\Scripts\python.exe')) {
        Invoke-Checked -Executable 'python' -Arguments @('-m', 'venv', '.venv')
    }
    $Python = '.\.venv\Scripts\python.exe'
    Invoke-Checked -Executable $Python -Arguments @('-m', 'pip', 'install', '-e', '.[dev,qlib,showcase]')
    if ($DownloadData -or -not (Test-Path 'data\qlib\cn_data\calendars\day.txt')) {
        Invoke-Checked -Executable $Python -Arguments @('scripts\download_sample_data.py')
    }
    Invoke-Checked -Executable $Python -Arguments @('-m', 'ruff', 'check', 'src', 'tests', 'scripts')
    Invoke-Checked -Executable $Python -Arguments @('-m', 'pytest', '--cov=qlib_factor_lab', '--cov-report=term-missing', '--cov-fail-under=80')
    Invoke-Checked -Executable $Python -Arguments @('scripts\run_portfolio_v2.py')
    Invoke-Checked -Executable $Python -Arguments @('scripts\build_showcase.py')
}
finally {
    Pop-Location
}
