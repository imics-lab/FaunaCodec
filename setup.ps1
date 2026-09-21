# Install FaunaCodec into a virtual environment and build the DCVC entropy-coder extension.
#
#   .\setup.ps1                          # venv in .\.venv, auto-detect CUDA
#   .\setup.ps1 -Cuda 12.4                # pin the PyTorch CUDA wheel
#   .\setup.ps1 -Venv C:\envs\fauna       # put the venv somewhere else
#   .\setup.ps1 -Python python3.11        # interpreter to build the venv from
#   .\setup.ps1 -NoVenv                   # install into the active environment
#   .\setup.ps1 -Extras upscale           # also install the diffusion upscaler stack
#   .\setup.ps1 -InstallCompiler          # winget-install the minimal MSVC toolset if missing (triggers one UAC prompt)
param(
    [string]$Cuda = "",
    [string]$Venv = "$PSScriptRoot\.venv",
    [string]$Python = "python",
    [string]$Extras = "metrics",
    [switch]$NoVenv,
    [switch]$InstallCompiler
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

# -- 0. PyTorch wheel index --------------------------------------------------
$Arch = $env:PROCESSOR_ARCHITECTURE
if ($Arch -ne "AMD64") {
    Write-Host "Detected ${Arch}: skipping the PyTorch wheel index."
    Write-Host "Install PyTorch from NVIDIA's channel for this platform before continuing."
    $TorchIndex = ""
} else {
    if (-not $Cuda) {
        if (Get-Command nvcc -ErrorAction SilentlyContinue) {
            $m = (nvcc --version | Select-String -Pattern 'release (\d+\.\d+)').Matches
            if ($m.Count -gt 0) { $Cuda = $m[0].Groups[1].Value }
        } elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            $m = (nvidia-smi | Select-String -Pattern 'CUDA Version: (\d+\.\d+)').Matches
            if ($m.Count -gt 0) { $Cuda = $m[0].Groups[1].Value }
        }
        if (-not $Cuda) {
            Write-Host "No CUDA toolkit found; installing CPU-only PyTorch."
            $Cuda = "cpu"
        }
    }
    switch -Regex ($Cuda) {
        "^cpu$"                    { $TorchIndex = "https://download.pytorch.org/whl/cpu" }
        "^12\.(6|7|8|9)$|^13\."    { $TorchIndex = "https://download.pytorch.org/whl/cu126" }
        "^12\.(4|5)$"              { $TorchIndex = "https://download.pytorch.org/whl/cu124" }
        default                    { $TorchIndex = "https://download.pytorch.org/whl/cu121" }
    }
}

$EnvLabel = if ($NoVenv) { "<active env>" } else { $Venv }
Write-Host "============================================================"
Write-Host " FaunaCodec setup"
Write-Host "   architecture: $Arch"
Write-Host "   CUDA:         $(if ($Cuda) { $Cuda } else { 'n/a' })"
Write-Host "   torch index:  $(if ($TorchIndex) { $TorchIndex } else { '<platform default>' })"
Write-Host "   environment:  $EnvLabel"
Write-Host "   extras:       $Extras"
Write-Host "============================================================"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "Warning: ffmpeg is not on PATH. The H.264/HEVC/AV1 backends and the lossless"
    Write-Host "         intermediates need it: 'winget install Gyan.FFmpeg' or 'choco install ffmpeg'."
}

# -- 1. Environment -----------------------------------------------------------
if (-not $NoVenv) {
    if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
        Write-Error "$Python not found; pass -Python."
        exit 1
    }
    & $Python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "FaunaCodec needs Python 3.10 or newer. Point setup.ps1 at one, e.g. -Python python3.11."
        exit 1
    }
    $VenvPython = Join-Path $Venv "Scripts\python.exe"
    if (Test-Path $VenvPython) {
        Write-Host "[1/4] Reusing the virtual environment in $Venv."
    } else {
        Write-Host "[1/4] Creating a virtual environment in $Venv ..."
        & $Python -m venv $Venv
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Creating the virtual environment failed; see the error above."
            exit 1
        }
    }
    $PyExe = $VenvPython
} else {
    Write-Host "[1/4] Using the active environment."
    $PyExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $PyExe) { $PyExe = (Get-Command python3 -ErrorAction Stop).Source }
}

# -- 2. PyTorch, then FaunaCodec -----------------------------------------------
Write-Host "[2/4] Installing dependencies ..."
& $PyExe -m pip install --upgrade pip
if ($TorchIndex) {
    & $PyExe -m pip install --index-url $TorchIndex torch torchvision
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
if ($Extras) {
    & $PyExe -m pip install -e "$Root[$Extras]"
} else {
    & $PyExe -m pip install -e $Root
}
if ($LASTEXITCODE -ne 0) { exit 1 }

# -- 3. DCVC entropy-coder extension -------------------------------------------
Write-Host "[3/4] Building the DCVC C++ entropy coder ..."
$DcvcCpp = Join-Path $Root "third_party\dcvc\src\cpp"
if (-not (Test-Path $DcvcCpp)) {
    Write-Error @"
DCVC is missing from $Root\third_party\dcvc.
It is vendored in this repository; re-clone FaunaCodec, or fetch it with:
  git clone https://github.com/microsoft/DCVC $Root\third_party\dcvc
"@
    exit 1
}

function Test-MsvcToolset {
    if (Get-Command cl -ErrorAction SilentlyContinue) { return $true }
    $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $VsWhere) {
        $vsPath = & $VsWhere -latest -products '*' `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath
        if ($vsPath) { return $true }
    }
    return $false
}

$HaveMsvc = Test-MsvcToolset

if (-not $HaveMsvc -and $InstallCompiler) {
    # A VS/Build Tools instance can already exist (just missing the C++ component). The
    # bootstrapper's "install" verb refuses whenever any instance is present, so an existing
    # instance must be grown with "modify --installPath", not "winget install". Only a
    # from-scratch machine (no instance at all) can go through winget's default install verb.
    $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $ExistingInstallPath = $null
    if (Test-Path $VsWhere) {
        # -all (not -latest): -latest silently excludes incomplete/canceled instances, and an
        # interrupted previous install is exactly the case this branch needs to catch.
        $paths = & $VsWhere -all -products '*' -property installationPath
        $ExistingInstallPath = $paths | Where-Object { $_ } | Select-Object -First 1
    }

    Write-Host "Installing the MSVC toolset + Windows SDK (~2-3 GB)."
    Write-Host "Windows will ask you to approve one UAC elevation prompt; the install itself is silent after that."

    if ($ExistingInstallPath) {
        $VsInstallerSetup = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\setup.exe"
        Write-Host "Found an existing instance at $ExistingInstallPath; adding components to it."
        # setup.exe refuses --passive/--quiet unless the CALLER is already elevated (it will not
        # self-elevate the way the winget/vs_BuildTools.exe bootstrapper does); Start-Process
        # -Verb RunAs is what actually raises the UAC prompt here. -Verb forces the ShellExecute
        # path, which (unlike CreateProcess) space-joins an -ArgumentList array with NO quoting,
        # so an unquoted $ExistingInstallPath gets truncated at its first space ("C:\Program
        # Files..." -> "C:\Program"). Build one pre-quoted string instead.
        $vsArgString = 'modify --installPath "' + $ExistingInstallPath + '"' +
            ' --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64' +
            ' --add Microsoft.VisualStudio.Component.Windows11SDK.22621' +
            ' --passive --norestart'
        try {
            $proc = Start-Process -FilePath $VsInstallerSetup -ArgumentList $vsArgString -Verb RunAs -Wait -PassThru
            $installExit = $proc.ExitCode
        } catch {
            Write-Host "UAC elevation was cancelled or failed: $($_.Exception.Message)"
            $installExit = 1
        }
    } else {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            Write-Error "winget not found; can't auto-install the compiler. Install App Installer from the Microsoft Store, or use the manual command below (drop -InstallCompiler)."
            exit 1
        }
        & winget install --id Microsoft.VisualStudio.2022.BuildTools --accept-package-agreements --accept-source-agreements `
            --override "--passive --norestart --wait --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
        $installExit = $LASTEXITCODE
    }

    # The modify/install exit code is not trustworthy on its own: setup.exe (unlike the
    # winget/vs_BuildTools.exe bootstrapper) can hand off to a background elevated worker and
    # return before that worker finishes, reporting exit 1 even though the components end up
    # installed correctly. vswhere's on-disk state is the source of truth, so always re-check
    # it regardless of what the exit code says.
    if ($installExit -ne 0 -and $installExit -ne 3010) {
        Write-Host "Compiler install step returned exit $installExit; re-checking actual state before giving up (this exit code alone isn't reliable here)."
    }
    $HaveMsvc = Test-MsvcToolset
}

if (-not $HaveMsvc) {
    Write-Error @'
No MSVC C++ toolset found. The vendored DCVC extension (third_party/dcvc/src/cpp/setup.py)
builds with MSVC on Windows, not MinGW/gcc. Install just the compiler and a Windows SDK
(about 2-3 GB, far less than the full "Desktop development with C++" workload) either by
re-running this script with -InstallCompiler (needs one UAC approval), or by hand from an
elevated PowerShell:

  & "C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe" modify `
    --installPath "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools" `
    --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    --add Microsoft.VisualStudio.Component.Windows11SDK.22621 `
    --passive --norestart

Then re-run setup.ps1. setuptools finds MSVC automatically once the component is installed;
no vcvarsall/Developer Prompt step is needed.
'@
    exit 1
}

& $PyExe -m pip install "pybind11>=2.11" "setuptools>=69"
Push-Location $DcvcCpp
try {
    & $PyExe -m pip install --no-build-isolation .
    if ($LASTEXITCODE -ne 0) { exit 1 }
} finally {
    Pop-Location
}

# -- 4. Verify ------------------------------------------------------------------
Write-Host "[4/4] Verifying ..."
$verify = @'
import sys

failures = []
for label, probe in [
    ("torch", lambda: __import__("torch")),
    ("opencv", lambda: __import__("cv2")),
    ("ultralytics", lambda: __import__("ultralytics")),
    ("faunacodec", lambda: __import__("faunacodec")),
    ("MLCodec_extensions_cpp", lambda: __import__("MLCodec_extensions_cpp")),
]:
    try:
        module = probe()
        version = getattr(module, "__version__", "ok")
        print(f"  {label:<24} {version}")
    except Exception as exc:
        failures.append(f"{label}: {exc}")

try:
    import torch
    print(f"  {'CUDA available':<24} {torch.cuda.is_available()}")
except Exception:
    pass

if failures:
    print("\nFailed:")
    for failure in failures:
        print(f"  {failure}")
    sys.exit(1)
print("\nAll imports OK.")
'@
$verify | & $PyExe -
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "Setup complete."
if (-not $NoVenv) { Write-Host "  $Venv\Scripts\Activate.ps1" }
Write-Host "  python scripts\download_models.py --group compress decompress"
Write-Host "  faunacodec-pipeline data\bird1.mp4 --stages compress decompress"
