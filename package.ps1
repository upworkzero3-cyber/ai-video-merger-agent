param([string]$Python = ".\.venv\Scripts\python.exe", [switch]$IncludeSpeech)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    $arguments = @(
        "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed", "--onedir",
        "--name", "AI Video Merger Agent", "--collect-data", "librosa",
        "--collect-data", "lazy_loader", "--collect-all", "numba",
        "--collect-all", "llvmlite", "--collect-submodules", "librosa",
        "--hidden-import", "scipy.signal", "--hidden-import", "sklearn.utils._cython_blas"
    )
    if ($IncludeSpeech) {
        $arguments += @("--collect-all", "faster_whisper", "--collect-all", "ctranslate2",
                        "--collect-all", "onnxruntime", "--collect-all", "tokenizers")
    }
    $arguments += "main.py"
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit code $LASTEXITCODE)." }
} finally {
    Pop-Location
}
