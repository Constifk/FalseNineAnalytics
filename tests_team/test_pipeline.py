"""Fast tests on synthetic data only (no devdata, no challenge data needed)."""
import numpy as np

from possession import methods, score
from possession.match import A, AWAY, D, H, HOME, X, Match, load_epts, write_submission


# ---- scorer ------------------------------------------------------------------------------------

def _gt():
    gt = np.full(1000, X, np.int8)
    gt[100:300] = H
    gt[300:350] = D
    gt[350:600] = A
    gt[600:900] = H
    return gt


def test_perfect_and_constant():
    gt = _gt()
    assert score.spell_f1(gt, gt) == 1.0
    assert score.spell_f1(gt, np.full(1000, D, np.int8)) == 0.0
    assert score.macro_f1(gt, gt) == 1.0


def test_one_frame_late_is_free():
    gt = _gt()
    late = np.roll(gt, 1)
    assert score.spell_f1(gt, late) == 1.0


def test_abstention_inside_possession_is_free_for_spell_f1():
    gt = _gt()
    pred = gt.copy()
    pred[150:250] = X
    assert score.spell_f1(gt, pred) == 1.0
    assert score.macro_f1(gt, pred) < 1.0


def test_flicker_is_punished():
    gt = _gt()
    pred = gt.copy()
    pred[620:625] = A  # one spurious possession splits a true one in two
    assert score.spell_f1(gt, pred) < 1.0


def test_halves_never_merge():
    gt = np.full(100, X, np.int8)
    gt[10:40] = H
    gt[60:90] = H
    assert score.valid_runs(gt) == [(10, 40), (60, 90)]
    assert len(score.spells(gt, score.valid_runs(gt))) == 2


# ---- EPTS loader ---------------------------------------------------------------------------------

META = """<?xml version="1.0"?>
<main><Metadata><Teams><Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away</Name></Team></Teams>
<Players><Player id="P1" teamId="T1"/><Player id="P2" teamId="T2"/></Players></Metadata></main>"""


def test_load_epts_roundtrip(tmp_path):
    (tmp_path / "m.xml").write_text(META)
    (tmp_path / "r.txt").write_text(
        "1:0.5,0.5,1.0;NaN,NaN,NaN:0.5,0.5,3.0\n"
        "2:0.6,0.5,1.0;0.1,0.2,0.0:NaN,NaN,NaN\n")
    m = load_epts(tmp_path / "m.xml", tmp_path / "r.txt", (100.0, 50.0), [(0, 2)], n_frames=3)
    assert m.px.shape == (3, 2)
    assert list(m.team) == [HOME, AWAY]
    assert m.px[0, 0] == 50.0 and np.isnan(m.px[0, 1])
    assert abs(m.px[1, 1] - 10.0) < 1e-4 and abs(m.py[1, 1] - 10.0) < 1e-4
    assert m.bx[0] == 50.0 and np.isnan(m.bx[1]) and np.isnan(m.bx[2])


def test_write_submission(tmp_path):
    p = write_submission(tmp_path / "t.csv", np.array([X, H, A, D]))
    assert open(p).read() == "frame,label\n0,X\n1,H\n2,A\n3,D\n"


# ---- decoder -------------------------------------------------------------------------------------

def test_viterbi_ignores_short_blips():
    em = np.zeros((200, 3))
    em[:, H] = 1.0
    em[100:103, A] = 3.0  # three frames of weak evidence for the other team
    path = methods.viterbi(em, methods.DecodeParams(beta=1.0, switch=8, stop=8, start=8))
    assert np.all(path == H)


def test_viterbi_follows_sustained_change():
    em = np.zeros((200, 3))
    em[:100, H] = 1.0
    em[100:, A] = 1.0
    path = methods.viterbi(em, methods.DecodeParams(beta=1.0, switch=8, stop=8, start=8))
    assert np.all(path[:100] == H) and np.all(path[100:] == A)


# ---- IDSSE loader --------------------------------------------------------------------------------

def test_idsse_parse_synthetic(tmp_path, monkeypatch):
    from possession import idsse

    monkeypatch.setattr(idsse, "_dir", lambda: tmp_path)
    mid = "J03WMX"
    (tmp_path / idsse.file_name("info", mid)).write_text(
        '<?xml version="1.0"?><PutDataRequest><MatchInformation>'
        '<Environment PitchX="105.00" PitchY="68.00"/><Teams>'
        '<Team TeamId="G" Role="guest"/><Team TeamId="Hm" Role="home"/>'
        '</Teams></MatchInformation></PutDataRequest>', encoding="utf-8")

    def frames(n0, n, extra=lambda i: ""):
        return "".join(f'<Frame N="{n0 + i}" T="t" X="{i * 0.1:.2f}" Y="-1.00" S="0.00"{extra(i)}/>'
                       for i in range(n))

    ball = lambda i: f' Z="0.1" M="1" BallPossession="{1 if i < 3 else 2}" BallStatus="{0 if i == 0 else 1}"'
    xml = ('<?xml version="1.0"?><PutDataRequest><Positions>'
           f'<FrameSet GameSection="firstHalf" MatchId="x" TeamId="referee" PersonId="R">{frames(10000, 5)}</FrameSet>'
           f'<FrameSet GameSection="firstHalf" MatchId="x" TeamId="Hm" PersonId="P1">{frames(10000, 5)}</FrameSet>'
           f'<FrameSet GameSection="firstHalf" MatchId="x" TeamId="G" PersonId="P2">{frames(10001, 4)}</FrameSet>'
           f'<FrameSet GameSection="firstHalf" MatchId="x" TeamId="BALL" PersonId="B">{frames(10000, 5, ball)}</FrameSet>'
           f'<FrameSet GameSection="secondHalf" MatchId="x" TeamId="BALL" PersonId="B">{frames(100000, 2, ball)}</FrameSet>'
           '</Positions></PutDataRequest>')
    (tmp_path / idsse.file_name("positions", mid)).write_text(xml, encoding="utf-8")

    m, gt = idsse.load(mid)
    assert m.runs == [(0, 5), (5 + idsse.HALF_GAP, 7 + idsse.HALF_GAP)]
    assert list(m.team) == [HOME, AWAY]                       # referee dropped, home first
    assert list(gt[:5]) == [D, H, H, A, A]
    assert gt[5] == X and list(gt[m.runs[1][0]:m.runs[1][1]]) == [D, H]
    assert np.isnan(m.px[0, 1]) and abs(m.px[2, 1] - (0.1 + 52.5)) < 1e-4   # centre -> corner origin
    assert abs(m.py[0, 0] - 33.0) < 1e-4
