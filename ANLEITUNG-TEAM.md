# Anleitung fürs Team (Yang, Consti, Sascha)

Stand 11.11.2026. Plan und Hintergrund: `yang-os/Work/more-than-a-hack/plan.md` (Yang schickt ihn rum).

## Die wichtigste Regel

**Die Challenge-Daten (`data/`) kommen in kein KI-Tool**: nicht in ChatGPT, nicht in Claude, Gemini,
Codex oder Cursor. Keine Dateien hochladen, keine Zeilen reinkopieren, keine Screenshots vom Video. Das
verbieten die Teilnahmebedingungen. Was ihr einer KI zeigen dürft: unseren Code und die **Zahlen**, die
die Skripte am Ende ausgeben (spellF1, macroF1, Anzahl Ballbesitzphasen).

Außerdem: Nichts im Spiel von Hand labeln oder korrigieren, und auf den öffentlichen 5 Minuten nicht
tunen. Video anschauen, um zu verstehen, ist erlaubt.

## Mit KI arbeiten ohne Daten

Ihr dürft mit Claude, ChatGPT, Codex oder Cursor am **Code** arbeiten. Die Grenze verläuft so:

| Darf zur KI | Darf nie zur KI |
|---|---|
| alles unter `possession/`, `tests_team/`, die `.md`-Dateien | alles in `data\` (ATD, Smart Tagging, Video, Homographie) |
| die offenen Testspiele in `devdata\` (Metrica, IDSSE/DFL) | alles in `out\` (unsere CSV ist aus den Daten abgeleitet) |
| die Textausgabe von `possession.run`: Phasenzahl, sha256, die drei Lader-Zeilen | Zeilen, Ausschnitte oder Spaltenköpfe aus diesen Dateien |
| Fehlermeldungen, **wenn** darin keine Datenzeilen stehen | Screenshots oder Clips aus dem Video, Plots vom echten Spiel |

So geht ihr praktisch vor:

1. **Code-Agent im Repo** (Claude Code, Codex, Cursor, Gemini CLI): Im Repo liegen Sperrdateien
   (`.claude/settings.json`, `.cursorignore`, `.geminiignore`, `.aiexclude`), die `data\` und `out\`
   ausschließen. Das ist ein Gurt, kein Schloss: Ein Agent mit Shell kann die Dateien trotzdem
   mit `type` oder `python` öffnen. Sagt ihm deshalb am Anfang ausdrücklich: „`data/` und `out/`
   nie öffnen, lesen oder ausgeben, auch nicht ausschnittsweise.“ Und lasst ihn **nie**
   `possession.run` oder andere Skripte auf `data\` ausführen, denn dann landet die Ausgabe bei ihm.
   Am sichersten liegen die Daten gar nicht im Repo: Paket z. B. nach `C:\mtah-daten` entpacken und
   vor dem Lauf `$env:MTAH_DATA_DIR = "C:\mtah-daten"` setzen. Dann sieht ein Agent im Repo nichts.
2. **Läufe auf den echten Daten macht ihr selbst** in PowerShell, ohne KI-Fenster daneben:
   `.venv\Scripts\python -m possession.run --team FalseNineAnalytics`. An die KI geht nur die
   Zeile mit Phasenzahl und sha256.
3. **Fehler auf den echten Daten:** Den Traceback nur weitergeben, wenn keine Datenzeile darin steht
   (bei Ladefehlern zeigt pandas manchmal eine Zeile an). Im Zweifel nur den Fehlertyp und die
   Codezeile abschreiben.
4. **Ideen testen** immer zuerst auf den offenen Testspielen: `python -m possession.bench --quick`.
   Das ist die einzige Messung, die wir der KI zeigen und auf der wir tunen dürfen.

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

Beim ersten Mal werden die offenen Testspiele geladen: 3 Metrica-Spiele (~200 MB) und 7 DFL-Spiele
(~2,6 GB, Hugging Face lädt manchmal langsam, also Zeit einplanen). Danach dauert ein Lauf ein paar Minuten.
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

| Version | Metrica (3 Spiele) | DFL (7 Spiele) | Idee |
|---|---|---|---|
| v0 | 0,29 | 0,25 | offizielle Baseline nachgebaut (offiziell auf dem echten Spiel: 0,243) |
| v1 | 0,38 | 0,32 | Regeln: Team behält den Ball, bis der Gegner ihn kontrolliert |
| v2 | 0,51 | 0,40 | Zustandsmodell H/A/D, trainiert auf Testspielen mit künstlichem Rauschen |

Neu gerechnet am 12.11. (`python -m possession.bench --quick`, jedes Spiel einmal als Test, trainiert
auf den anderen neun, Laufzeit ~50 Min.). Die DFL-Spiele sind der ehrlichere Maßstab: Die Baseline
landet dort bei 0,25, fast genau wie offiziell auf dem echten Spiel. Größte Schwäche: v2 findet auf
den DFL-Spielen nur 59 % der echten Ballbesitzphasen.

Ob das auf dem echten Spiel genauso aussieht, wissen wir erst nach der Abgabe. Es gibt keine
Live-Rangliste.

## Ideen für Sonntag (ohne Programmieren)

Schaut euch im Taktikvideo die ersten 5 Minuten mit der öffentlichen Musterlösung an (Frame 11185
bis 18684, 25 Frames = 1 Sekunde). `python -m possession.srt` legt sie als Untertitel neben das Video
(`data\tactical\BAR-ATH_tactical.srt`), VLC zeigt sie automatisch an. Mit Strg+T zu 00:07:27 springen,
mit E geht es ein Bild weiter. Schreibt **allgemeine Regeln** auf, keine Frame-Nummern:

1. Wann beginnt ein toter Ball genau: wenn der Ball über die Linie geht oder beim Pfiff?
2. Wann endet er: wenn der Standard ausgeführt wird, oder schon vorher?
3. Wo genau wechselt der Ballbesitz: bei der ersten Berührung des Gegners oder erst bei Kontrolle?
4. Was passiert bei Abprallern, Kopfballduellen und geblockten Pässen?
