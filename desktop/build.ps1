<#
ccPlotter desktop build: stamps desktop/version.json (commit sha, branch,
build date, dirty flag) from the current git state, then runs PyInstaller
via desktop/ccplotter.spec. version.json is a regenerated build artifact --
not committed -- so it always reflects the tree that actually got built,
never a stale checked-in value.

Usage (from anywhere):
    powershell -File desktop\build.ps1
#>
$ErrorActionPreference = "Stop"

$desktopDir = $PSScriptRoot
$root = Split-Path -Parent $desktopDir
Set-Location $root

$commit = (git rev-parse --short HEAD).Trim()
$branch = (git rev-parse --abbrev-ref HEAD).Trim()

# Dirty = tracked files with uncommitted changes (staged or unstaged). Ignore
# untracked files -- stray untracked files do not
# mean the sources that get bundled are dirty.
$trackedChanges = git status --porcelain --untracked-files=no
$dirty = [bool]($trackedChanges -and $trackedChanges.Trim().Length -gt 0)

$buildDate = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

$versionObj = [ordered]@{
    commit     = $commit
    branch     = $branch
    build_date = $buildDate
    dirty      = $dirty
}
$versionJson = $versionObj | ConvertTo-Json -Compress
$versionPath = Join-Path $desktopDir "version.json"
Set-Content -Path $versionPath -Value $versionJson -Encoding ascii -NoNewline

Write-Host "[build] $versionPath"
Write-Host "[build] $versionJson"
if ($dirty) {
    Write-Host "[build] WARNING: tracked working tree has uncommitted changes -- dirty=true" -ForegroundColor Yellow
}

# Invoke via `python -m PyInstaller` rather than the bare `pyinstaller` shim --
# the shim isn't guaranteed to be on PATH, while the
# module always is if PyInstaller is installed in the active interpreter.
python -m PyInstaller (Join-Path $desktopDir "ccplotter.spec") --noconfirm
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

Write-Host "[build] Done. Output: $(Join-Path $root 'dist\ccPlotter')"
