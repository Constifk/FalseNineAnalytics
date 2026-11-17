# Erzeugt die Abgabe out\FalseNineAnalytics.csv auf diesem Rechner (Windows).
# Start per Doppelklick auf ABGABE-ERSTELLEN.bat.
#
# Die Challenge-Daten bleiben auf diesem Rechner. Das Protokoll ABGABE-LOG.txt enthaelt nur
# Zusammenfassungen (Anteil Ball sichtbar, Anzahl Ballbesitzphasen, sha256, Pruefergebnis),
# keine Datenzeilen. Nur dieses Protokoll darf an eine KI weitergegeben werden.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Team = "FalseNineAnalytics"
Start-Transcript -Path (Join-Path $Root "ABGABE-LOG.txt") -Force | Out-Null

function Step($t) { Write-Host ""; Write-Host "=== $t ===" -ForegroundColor Cyan }

try {
    # 1. Python 3.12 finden oder installieren
    Step "1/6 Python 3.12 pruefen"
    $Py = $null
    try { & py -3.12 --version *> $null; if ($LASTEXITCODE -eq 0) { $Py = @("py", "-3.12") } } catch {}
    if (-not $Py) {
        $cand = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
        if (-not (Test-Path $cand)) {
            Write-Host "Python 3.12 nicht gefunden, installiere es mit winget ..."
            winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
        }
        if (-not (Test-Path $cand)) { throw "Python 3.12 konnte nicht installiert werden. Bitte von python.org installieren und erneut starten." }
        $Py = @($cand)
    }
    $PyExe = $Py[0]; $PyArgs = @($Py | Select-Object -Skip 1)
    & $PyExe @PyArgs --version

    # 2. Daten einsortieren
    Step "2/6 Daten einsortieren"
    $atdTxt = Join-Path $Root "data\tactical\BAR-ATH_tactical_atd.txt"
    $atdXml = Join-Path $Root "data\tactical\BAR-ATH_tactical_atd.xml"
    $inbox = Join-Path $Root "DATEN-HIER-REIN"
    if (-not ((Test-Path $atdTxt) -and (Test-Path $atdXml))) {
        $items = @(Get-ChildItem $inbox -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne "LIESMICH.txt" })
        if ($items.Count -eq 0) {
            throw "Keine Daten gefunden. Lege die Dateien aus dem Google-Drive-Ordner (oder die heruntergeladenen ZIPs) in den Ordner DATEN-HIER-REIN und starte erneut."
        }
        & $PyExe @PyArgs -m possession.unpack @($items | ForEach-Object { $_.FullName })
    }
    if (-not ((Test-Path $atdTxt) -and (Test-Path $atdXml))) {
        throw "Es fehlen BAR-ATH_tactical_atd.txt und/oder BAR-ATH_tactical_atd.xml. Beide aus dem Ordner 'tactical' im Drive herunterladen."
    }
    Write-Host "Tracking-Dateien gefunden."

    # 3. Umgebung
    Step "3/6 Umgebung einrichten (.venv)"
    if (-not (Test-Path ".venv\Scripts\python.exe")) { & $PyExe @PyArgs -m venv .venv }
    $V = Join-Path $Root ".venv\Scripts\python.exe"
    & $V -m pip install --disable-pip-version-check -q -r requirements-team.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install fehlgeschlagen" }

    # 4. Abgabe erzeugen (laedt beim ersten Mal ~200 MB offene Trainingsspiele)
    Step "4/6 Abgabe erzeugen (einige Minuten)"
    & $V -m possession.run --team $Team --train metrica
    if ($LASTEXITCODE -ne 0) { throw "possession.run fehlgeschlagen" }

    # 5. Offizielle Pruefung
    Step "5/6 Offizielle Formatpruefung"
    & $V evaluation\validate_submission.py "out\$Team.csv"
    if ($LASTEXITCODE -ne 0) { throw "Die Datei hat die offizielle Pruefung NICHT bestanden" }

    # 6. Dateien zum Hochladen bereitlegen
    Step "6/6 Dateien zum Hochladen bereitlegen"
    $Up = Join-Path $Root "ZUM-HOCHLADEN"
    New-Item -ItemType Directory -Force -Path $Up | Out-Null
    Copy-Item "out\$Team.csv" (Join-Path $Up "$Team.csv") -Force
    if (Test-Path "$Team.pdf") { Copy-Item "$Team.pdf" (Join-Path $Up "$Team.pdf") -Force }
    $h = (Get-FileHash (Join-Path $Up "$Team.csv") -Algorithm SHA256).Hash
    Write-Host "sha256 $Team.csv: $h"
    Write-Host ""
    Write-Host "FERTIG. Im Ordner ZUM-HOCHLADEN liegen $Team.csv und $Team.pdf." -ForegroundColor Green
}
catch {
    Write-Host ""
    Write-Host "FEHLER: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    Stop-Transcript | Out-Null
    Write-Host ""
    Write-Host "Protokoll: ABGABE-LOG.txt. Fenster mit Enter schliessen."
    [void](Read-Host)
}
