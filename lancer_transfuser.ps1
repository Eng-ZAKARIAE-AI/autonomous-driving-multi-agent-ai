Set-Location $PSScriptRoot

# Activer venv37 (Python 3.7 — carla + torch + timm)
& "$PSScriptRoot\venv37\Scripts\Activate.ps1"

Write-Host ""
Write-Host "Python : $(python --version)"
Write-Host "Venv   : $env:VIRTUAL_ENV"
Write-Host ""

# Verifier CARLA avant de lancer
Write-Host "--- Verification CARLA ---"
python "$PSScriptRoot\check_carla.py"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "CARLA non pret. Lance CarlaUE4.exe et reessaie." -ForegroundColor Red
    Read-Host "Appuie sur Entree pour quitter"
    exit 1
}

Write-Host ""
Write-Host "--- Lancement TransFuser ---"
python "$PSScriptRoot\run_transfuser.py"
