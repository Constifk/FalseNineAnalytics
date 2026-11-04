# Anleitung fürs Team (Yang, Consti, Sascha)

Stand 04.11.2026. Plan und Hintergrund: `yang-os/Work/more-than-a-hack/plan.md` (Yang schickt ihn rum).

## Die wichtigste Regel

**Die Challenge-Daten (`data/`) kommen in kein KI-Tool**: nicht in ChatGPT, nicht in Claude, Gemini,
Codex oder Cursor. Keine Dateien hochladen, keine Zeilen reinkopieren, keine Screenshots vom Video. Das
verbieten die Teilnahmebedingungen. Was ihr einer KI zeigen dürft: unseren Code und die **Zahlen**, die
die Skripte am Ende ausgeben (spellF1, macroF1, Anzahl Ballbesitzphasen).

Außerdem: Nichts im Spiel von Hand labeln oder korrigieren, und auf den öffentlichen 5 Minuten nicht
tunen. Video anschauen, um zu verstehen, ist erlaubt.

## Einmalig einrichten (Windows, PowerShell, im Ordner `mtah-2027`)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-team.txt
```

Datenpaket aus der Mail nach `data\` entpacken, so wie in `data\README.md` beschrieben:
`data\tactical\BAR-ATH_tactical_atd.txt` usw.

## Abgabe erzeugen

```powershell
.venv\Scripts\python -m possession.run --team TEAMNAME
.venv\Scripts\python evaluation\validate_submission.py out\TEAMNAME.csv
```

Das dauert ein paar Minuten. Beim ersten Mal werden die offenen Testspiele (~200 MB) geladen.
Die Ausgabe (nur die Textzeilen) an Jarvis schicken. Die ersten drei Zeilen prüfen den Lader:
Ball 59,3 % und 20,5 Spieler sind die offiziellen Werte aus DATA.md. Weicht es stark ab, stimmt
etwas mit dem Einlesen nicht.

Zum Vergleich die offizielle Baseline:

```powershell
.venv\Scripts\python starter\baseline.py
.venv\Scripts\python evaluation\run_evaluation.py
```

Sie muss auf den 5 öffentlichen Minuten so abschneiden, wie es in EVALUATION.md steht
(`baseline_smooth50`: spellF1 0,3256). Dann wissen wir, dass Daten und Scorer stimmen.

## Wo wir stehen (gemessen nur auf offenen Testspielen, nicht auf dem Challenge-Spiel)

| Version | spellF1 Testspiele | Idee |
|---|---|---|
| v0 | 0,26 | offizielle Baseline nachgebaut (offiziell auf dem echten Spiel: 0,243) |
| v1 | 0,39 | Regeln: Team behält den Ball, bis der Gegner ihn kontrolliert |
| v2 | 0,51 | Zustandsmodell H/A/D, trainiert auf Testspielen mit künstlichem Rauschen |

Ob das auf dem echten Spiel genauso aussieht, wissen wir erst nach der Abgabe. Es gibt keine
Live-Rangliste.

## Ideen für Sonntag (ohne Programmieren)

Schaut euch im Taktikvideo die ersten 5 Minuten neben `evaluation/gt/BAR-ATH_possession_gt_public.csv`
an (Frame 11185 bis 18684, 25 Frames = 1 Sekunde) und schreibt **allgemeine Regeln** auf, keine
Frame-Nummern:

1. Wann beginnt ein toter Ball genau: wenn der Ball über die Linie geht oder beim Pfiff?
2. Wann endet er: wenn der Standard ausgeführt wird, oder schon vorher?
3. Wo genau wechselt der Ballbesitz: bei der ersten Berührung des Gegners oder erst bei Kontrolle?
4. Was passiert bei Abprallern, Kopfballduellen und geblockten Pässen?
