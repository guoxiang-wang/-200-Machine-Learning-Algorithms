param(
  [switch]$Serve,
  [string]$PythonPath
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot

try {
  if (-not $PythonPath) {
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($conda) {
      $condaRoot = (& $conda.Source info --base).Trim()
      $condaPython = Join-Path $condaRoot 'python.exe'
      if (Test-Path -LiteralPath $condaPython) {
        $PythonPath = $condaPython
      }
    }
  }

  if (-not $PythonPath) {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
      $PythonPath = $python.Source
    }
  }

  if (-not $PythonPath) {
    throw 'Python was not found. Pass -PythonPath or activate a Python environment.'
  }

  if ($Serve) {
    & $PythonPath -m mkdocs serve
  }
  else {
    & $PythonPath -m mkdocs build --strict
  }

  if ($LASTEXITCODE -ne 0) {
    throw "MkDocs exited with code $LASTEXITCODE."
  }
}
finally {
  Pop-Location
}
