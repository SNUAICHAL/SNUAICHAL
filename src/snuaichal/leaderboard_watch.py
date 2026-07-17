"""Resilient terminal watcher for the SNU AI Challenge public leaderboard."""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


DEFAULT_COMPETITION = "snuaichallenge"
DEFAULT_TEAM = "밥먹을돈으로3090사서거지됨"
DEFAULT_INTERVAL = 30.0
DEFAULT_PROJECTED_CUT = 0.95
DEFAULT_SHOW = 20
KST = timezone(timedelta(hours=9))
COMPETITION_START = datetime(2026, 6, 29, 10, 0, tzinfo=KST)
COMPETITION_DEADLINE = datetime(2026, 7, 24, 23, 59, tzinfo=KST)
PROGRESS_WIDTH = 32

# (last rank, label, color). The competition awards seven teams and reviews
# code from the top sixteen. TOP10 is retained as a useful final-round marker.
TIERS = (
    (7, "최종 수상 예상", "gold"),
    (10, "본선 진출 예상", "green"),
    (16, "코드 검증 예상", "blue"),
)
AWARD_N = TIERS[0][0]

RANK_W, NAME_W, SCORE_W, DATE_W = 4, 30, 10, 19
LINE_W = 86
RULE = "-" * (LINE_W - 2)
ANSI = re.compile(r"\x1b\[[0-9;]*m")

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[94m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "gold": "\033[93m",
    "onus": "\033[45m\033[97m\033[1m",
}

COLOR_ENABLED = True


class LeaderboardError(RuntimeError):
    """Raised when a leaderboard response cannot safely become a snapshot."""


@dataclass(frozen=True)
class Credentials:
    """A ready-to-send authorization header whose secret is never repr'd."""

    authorization: str = field(repr=False)
    source: str


@dataclass(frozen=True)
class CompetitionProgress:
    """A clamped snapshot of the preliminary-round competition clock."""

    phase: str
    ratio: float
    elapsed: timedelta
    remaining: timedelta


def _nonempty(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def load_credentials(
    *,
    environ: Mapping[str, str] | None = None,
    config_path: Path | None = None,
) -> Credentials:
    """Load Kaggle credentials without ever logging their values.

    A modern standalone token may be supplied as ``KAGGLE_API_TOKEN``. The
    classic Kaggle credentials use HTTP Basic auth and therefore require both
    username and key, from environment variables or ``kaggle.json``.
    """

    env = os.environ if environ is None else environ
    bearer = _nonempty(env.get("KAGGLE_API_TOKEN"))
    if bearer:
        return Credentials(f"Bearer {bearer}", "KAGGLE_API_TOKEN")

    path = config_path or Path(
        env.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
    ) / "kaggle.json"
    file_data: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LeaderboardError(f"Kaggle credential file is invalid: {path}") from exc
        if isinstance(loaded, dict):
            file_data = loaded

    username = _nonempty(env.get("KAGGLE_USERNAME")) or _nonempty(
        str(file_data.get("username", ""))
    )
    key = _nonempty(env.get("KAGGLE_KEY")) or _nonempty(
        str(file_data.get("key", ""))
    )
    if not username or not key:
        raise LeaderboardError(
            "Kaggle credentials were not found. Set KAGGLE_USERNAME and "
            "KAGGLE_KEY together, set KAGGLE_API_TOKEN, or install "
            f"kaggle.json at {path}."
        )
    encoded = base64.b64encode(f"{username}:{key}".encode("utf-8")).decode("ascii")
    source = "environment/config" if env.get("KAGGLE_USERNAME") or env.get("KAGGLE_KEY") else str(path)
    return Credentials(f"Basic {encoded}", source)


def parse_submission_time(value: str) -> datetime:
    """Parse Kaggle's ISO timestamp as an aware UTC datetime."""

    text = str(value).strip()
    if not text:
        raise LeaderboardError("A leaderboard row has no submission timestamp")
    # Python 3.10 accepts microseconds, while Kaggle can emit seven fractional
    # digits. Truncate (rather than round) only the unsupported tail.
    text = re.sub(r"(\.\d{6})\d+(?=Z|[+-]\d{2}:?\d{2}|$)", r"\1", text)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        # Some responses use a space separator and seven fractional digits.
        match = re.match(
            r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$",
            str(value),
        )
        if not match:
            raise LeaderboardError(f"Invalid submission timestamp: {value!r}")
        parsed = datetime.strptime(" ".join(match.groups()), "%Y-%m-%d %H:%M:%S")
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_kst(value: str) -> str:
    return parse_submission_time(value).astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")


def parse_schedule_time(value: str) -> datetime:
    """Parse a CLI schedule value and default timezone-less input to KST."""

    text = value.strip().replace(" ", "T")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected an ISO datetime such as 2026-07-24T23:59+09:00"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KST)
    return parsed.astimezone(KST)


def competition_progress(
    at: datetime,
    *,
    start: datetime = COMPETITION_START,
    deadline: datetime = COMPETITION_DEADLINE,
) -> CompetitionProgress:
    """Return preliminary-round progress before, during, or after the event."""

    if start.tzinfo is None or deadline.tzinfo is None or at.tzinfo is None:
        raise ValueError("competition clock datetimes must be timezone-aware")
    if deadline <= start:
        raise ValueError("competition deadline must be later than its start")
    if at < start:
        return CompetitionProgress("upcoming", 0.0, timedelta(0), start - at)
    if at >= deadline:
        return CompetitionProgress("finished", 1.0, deadline - start, timedelta(0))
    total = deadline - start
    elapsed = at - start
    return CompetitionProgress("running", elapsed / total, elapsed, deadline - at)


def format_duration(value: timedelta, *, include_seconds: bool = True) -> str:
    """Format a non-negative duration compactly in Korean."""

    total_seconds = max(0, int(value.total_seconds()))
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}일")
    if days or hours:
        parts.append(f"{hours}시간")
    if days or hours or minutes:
        parts.append(f"{minutes}분")
    if include_seconds or not parts:
        parts.append(f"{seconds}초")
    return " ".join(parts)


def competition_status_text(
    *,
    at: datetime | None = None,
    start: datetime = COMPETITION_START,
    deadline: datetime = COMPETITION_DEADLINE,
    compact: bool = False,
) -> str:
    """Build the live progress/countdown text used by table and heartbeat."""

    current = datetime.now(KST) if at is None else at.astimezone(KST)
    progress = competition_progress(current, start=start, deadline=deadline)
    percent = progress.ratio * 100
    if progress.phase == "upcoming":
        return f"예선 시작까지 {format_duration(progress.remaining)}"
    if progress.phase == "finished":
        return "예선 종료 · 최종 리더보드 공개 대기"
    remaining = format_duration(progress.remaining)
    if compact:
        return f"예선 {percent:5.2f}% · 마감까지 {remaining}"
    filled = min(PROGRESS_WIDTH, max(0, round(progress.ratio * PROGRESS_WIDTH)))
    bar = "█" * filled + "░" * (PROGRESS_WIDTH - filled)
    return f"[{bar}] {percent:6.2f}% · 남은 시간 {remaining}"


def _page_url(api_url: str, page_token: str | None) -> str:
    if not page_token:
        return api_url
    separator = "&" if "?" in api_url else "?"
    return api_url + separator + urllib.parse.urlencode({"pageToken": page_token})


def fetch_leaderboard(
    credentials: Credentials,
    *,
    competition: str = DEFAULT_COMPETITION,
    timeout: float = 15.0,
    excluded_teams: Sequence[str] = (),
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, dict[str, Any]]:
    """Fetch every leaderboard page and return one ranked row per team."""

    api_url = f"https://www.kaggle.com/api/v1/competitions/{competition}/leaderboard/view"
    excluded = {unicodedata.normalize("NFC", value) for value in excluded_teams}
    rows: dict[str, dict[str, Any]] = {}
    page: str | None = None
    seen_pages: set[str] = set()

    while True:
        url = _page_url(api_url, page)
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": credentials.authorization,
                "Accept": "application/json",
                "Cache-Control": "no-cache, no-store",
                "User-Agent": "snuaichal-leaderboard-watch/1.0",
            },
        )
        with opener(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        submissions = payload.get("submissions", [])
        if not isinstance(submissions, list):
            raise LeaderboardError("Kaggle response has no submissions list")

        for submission in submissions:
            if not isinstance(submission, dict):
                continue
            team = unicodedata.normalize("NFC", str(submission.get("teamName", "")).strip())
            if not team or team in excluded or team in rows:
                continue
            try:
                score = float(submission["score"])
                updated_at = to_kst(str(submission["submissionDate"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise LeaderboardError(f"Invalid leaderboard row for team {team!r}") from exc
            rows[team] = {
                "team": team,
                "rank": len(rows) + 1,
                "score": score,
                "updated_at": updated_at,
            }

        next_page = payload.get("nextPageToken")
        if not next_page:
            break
        page = str(next_page)
        if page in seen_pages:
            raise LeaderboardError("Kaggle returned a repeated pagination token")
        seen_pages.add(page)

    if not rows:
        raise LeaderboardError("Kaggle returned an empty leaderboard; snapshot was not replaced")
    return rows


def normalize_rows(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate persisted rows and restore deterministic rank order."""

    ordered = sorted(rows.values(), key=lambda row: int(row["rank"]))
    normalized: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(ordered, start=1):
        team = unicodedata.normalize("NFC", str(row["team"]))
        normalized[team] = {
            "team": team,
            "rank": index,
            "score": float(row["score"]),
            "updated_at": str(row["updated_at"]),
        }
    return normalized


def save_snapshot(
    path: Path,
    *,
    competition: str,
    rows: Mapping[str, Mapping[str, Any]],
) -> None:
    """Atomically persist a secret-free snapshot for restart-safe diffs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "competition": competition,
        "fetched_at": datetime.now(KST).isoformat(),
        "rows": list(sorted(rows.values(), key=lambda row: int(row["rank"]))),
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_snapshot(path: Path, *, competition: str) -> dict[str, dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("competition") != competition:
            return None
        rows = payload["rows"]
        if not isinstance(rows, list):
            raise TypeError("rows is not a list")
        return normalize_rows({str(row["team"]): row for row in rows})
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LeaderboardError(f"Saved leaderboard state is invalid: {path}") from exc


def rank_cut(rows: Mapping[str, Mapping[str, Any]], rank: int) -> tuple[float | None, str | None]:
    for row in rows.values():
        if int(row["rank"]) == rank:
            return float(row["score"]), str(row["team"])
    return None, None


def diff_events(
    previous: Mapping[str, Mapping[str, Any]],
    current: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return structured changes, including score/rank and tier-cut movement."""

    events: list[dict[str, Any]] = []
    for team, row in current.items():
        old = previous.get(team)
        if old is None:
            events.append({"kind": "new", "team": team, "rank": row["rank"], "score": row["score"]})
            continue
        if abs(float(old["score"]) - float(row["score"])) > 1e-9:
            events.append(
                {
                    "kind": "score",
                    "team": team,
                    "old_score": old["score"],
                    "score": row["score"],
                    "delta": float(row["score"]) - float(old["score"]),
                    "updated_at": row["updated_at"],
                }
            )
        if int(old["rank"]) != int(row["rank"]):
            events.append(
                {
                    "kind": "rank",
                    "team": team,
                    "old_rank": old["rank"],
                    "rank": row["rank"],
                }
            )

    for team in previous.keys() - current.keys():
        events.append({"kind": "removed", "team": team})

    for rank, label, _ in TIERS:
        old_cut, _ = rank_cut(previous, rank)
        new_cut, holder = rank_cut(current, rank)
        if old_cut is None or new_cut is None or abs(old_cut - new_cut) <= 1e-9:
            continue
        events.append(
            {
                "kind": "cut",
                "rank": rank,
                "label": label,
                "old_score": old_cut,
                "score": new_cut,
                "delta": new_cut - old_cut,
                "holder": holder,
            }
        )
    return events


def append_events(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    observed_at = datetime.now(KST).isoformat()
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps({"observed_at": observed_at, **event}, ensure_ascii=False) + "\n")


def width(value: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in value)


def fit(value: str, target: int) -> str:
    if width(value) <= target:
        return value + " " * (target - width(value))
    output, used = "", 0
    for char in value:
        char_width = width(char)
        if used + char_width > target - 1:
            break
        output += char
        used += char_width
    return output + "…" + " " * (target - used - 1)


def paint(value: str, *styles: str) -> str:
    if not COLOR_ENABLED:
        return value
    return "".join(COLORS[style] for style in styles) + value + COLORS["reset"]


def visible_width(value: str) -> int:
    return width(ANSI.sub("", value))


def right_align(value: str, target: int) -> str:
    return " " * max(0, target - width(value)) + value


def spread(left: str, right: str, total: int = LINE_W) -> str:
    return left + " " * max(1, total - visible_width(left) - visible_width(right)) + right


def tier_of(rank: int) -> tuple[int, str, str] | None:
    for tier in TIERS:
        if rank <= tier[0]:
            return tier
    return None


def _cutline(projected_cut: float) -> str:
    prefix = f"  ═══ Projected Final TOP{AWARD_N} Cutoff "
    score_end = 2 + RANK_W + 1 + NAME_W + 1 + SCORE_W
    body = prefix + "═" * max(0, score_end - SCORE_W - 1 - visible_width(prefix))
    return paint(f"{body} {projected_cut:>{SCORE_W}.5f}", "cyan", "bold")


def _tier_rule(tier: tuple[int, str, str]) -> str:
    rank, label, color = tier
    prefix = f"  └─ TOP{rank} {label} 컷 "
    return paint(prefix + "─" * max(0, len(RULE) + 2 - width(prefix)), color)


def render_table(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    our_team: str,
    show: int,
    projected_cut: float,
    competition_start: datetime = COMPETITION_START,
    competition_deadline: datetime = COMPETITION_DEADLINE,
    current_time: datetime | None = None,
) -> None:
    ordered = sorted(rows.values(), key=lambda row: int(row["rank"]))
    shown: list[Mapping[str, Any] | int] = list(ordered if show == 0 else ordered[:show])
    our_team = unicodedata.normalize("NFC", our_team)
    us = rows.get(our_team)
    hidden = ordered[len(shown) :]
    if hidden:
        if us and int(us["rank"]) > len(shown):
            shown.extend([max(0, len(hidden) - 1), us])
        else:
            shown.append(len(hidden))

    now_kst = datetime.now(KST) if current_time is None else current_time.astimezone(KST)
    progress = competition_progress(
        now_kst, start=competition_start, deadline=competition_deadline
    )
    progress_color = (
        "red"
        if progress.phase == "running" and progress.remaining <= timedelta(days=1)
        else "cyan"
    )
    print()
    print(paint("  예선 대회 진행도", "bold"))
    print(
        "  "
        + paint(
            competition_status_text(
                at=now_kst,
                start=competition_start,
                deadline=competition_deadline,
            ),
            progress_color,
            "bold",
        )
    )
    print(
        paint(
            "  "
            f"{competition_start.astimezone(KST):%Y-%m-%d %H:%M} → "
            f"{competition_deadline.astimezone(KST):%Y-%m-%d %H:%M} KST",
            "dim",
        )
    )
    print()
    legend = "  ".join(
        paint(f"■ TOP{rank} {label}", color, "bold") for rank, label, color in TIERS
    )
    print(f"  {legend}")
    print(
        paint(
            f"  {fit('순위', RANK_W)} {fit('팀명', NAME_W)} "
            f"{right_align('Public LB', SCORE_W)}  최종 제출 (KST)",
            "bold",
        )
    )
    print(paint("  " + RULE, "dim"))

    cut_drawn = False
    for row in shown:
        if isinstance(row, int):
            print(paint(f"  {fit('⋮', RANK_W)} 이하 {row}팀 생략", "dim"))
            continue
        if not cut_drawn and float(row["score"]) < projected_cut:
            print(_cutline(projected_cut))
            cut_drawn = True
        line = (
            f"  {int(row['rank']):<{RANK_W}} {fit(str(row['team']), NAME_W)} "
            f"{float(row['score']):>{SCORE_W}.5f}  {row['updated_at']}"
        )
        tier = tier_of(int(row["rank"]))
        if row["team"] == our_team:
            print(paint("▶" + line[1:], "onus"))
        elif tier:
            print(paint(line, tier[2]))
        else:
            print(line)
        if tier and int(row["rank"]) == tier[0]:
            print(_tier_rule(tier))
    if not cut_drawn:
        print(_cutline(projected_cut))
    print(paint("  " + RULE, "dim"))

    our_score = float(us["score"]) if us else None

    def gap(target: float) -> str:
        if our_score is None:
            return ""
        delta = our_score - target
        return "→ 우리 " + paint(f"{delta:+.5f}", "green" if delta >= 0 else "red")

    summary: list[tuple[float, str, str, str]] = []
    for rank, label, color in TIERS:
        score, holder = rank_cut(rows, rank)
        if score is not None:
            summary.append(
                (score, f"현재 TOP{rank:<2} 컷 ({label})", paint(f"{score:.5f}", color, "bold"), f"({holder})")
            )
    cleared = sum(float(row["score"]) >= projected_cut for row in rows.values())
    summary.append(
        (
            projected_cut,
            f"Projected Final TOP{AWARD_N} Cutoff",
            paint(f"{projected_cut:.5f}", "cyan", "bold"),
            paint(f"(현재 통과 {cleared}팀)", "dim"),
        )
    )
    label_width = max(width(item[1]) for item in summary)
    for score, label, score_text, note in sorted(summary, key=lambda item: -item[0]):
        print(
            spread(
                f"  {label}{' ' * (label_width - width(label))} : {score_text} {note}",
                gap(score),
            )
        )

    if us:
        tier = tier_of(int(us["rank"]))
        status = paint(tier[1], tier[2], "bold") if tier else paint("권외", "red", "bold")
        print(
            f"  {paint(our_team, 'magenta', 'bold')} : "
            f"{paint(str(us['rank']) + '위', 'bold')} / {len(rows)}팀 "
            f"(Public LB {our_score:.5f}) · {status}"
        )
    else:
        print(paint(f"  우리 팀을 찾지 못했습니다: {our_team!r}", "yellow", "bold"))
        print(paint("  --team 값과 Kaggle 팀명의 공백·철자를 확인하세요.", "dim"))
    print(flush=True)


def format_event(event: Mapping[str, Any], *, our_team: str) -> str:
    kind = event["kind"]
    team = str(event.get("team", ""))
    ours = paint("★ 우리팀", "magenta", "bold") + " " if team == our_team else ""
    if kind == "new":
        return f"{paint('＋ 신규', 'cyan', 'bold')} {paint(team, 'bold')} {event['rank']}위 진입 (Public LB {event['score']:.5f})"
    if kind == "removed":
        return f"{paint('－ 이탈', 'dim')} {team}"
    if kind == "score":
        delta = float(event["delta"])
        arrow = paint(f"{'▲' if delta > 0 else '▼'}{delta:+.5f}", "green" if delta > 0 else "red")
        return f"{paint('점수', 'yellow')} {ours}{paint(team, 'bold')} {event['old_score']:.5f} → {event['score']:.5f} {arrow} {paint('(' + str(event['updated_at']) + ')', 'dim')}"
    if kind == "rank":
        improved = int(event["rank"]) < int(event["old_rank"])
        arrow = paint(
            f"{'↑' if improved else '↓'}{event['old_rank']}→{event['rank']}위",
            "green" if improved else "red",
        )
        return f"{paint('순위', 'blue')} {ours}{paint(team, 'bold')} {arrow}"
    if kind == "cut":
        delta = float(event["delta"])
        arrow = paint(f"{'▲' if delta > 0 else '▼'}{delta:+.5f}", "green" if delta > 0 else "red")
        label = paint(f"TOP{event['rank']} 컷({event['label']})", "cyan", "bold")
        return (
            f"{label} {event['old_score']:.5f} → {event['score']:.5f} {arrow} "
            f"(현재 {event['rank']}위: {event['holder']})"
        )
    return json.dumps(dict(event), ensure_ascii=False)


def _enable_windows_ansi() -> None:
    if sys.platform != "win32" or not COLOR_ENABLED:
        return
    try:
        import ctypes

        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except (AttributeError, OSError):
        pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--max-backoff", type=float, default=300.0)
    parser.add_argument("--table", action="store_true", help="Print the table even without changes")
    parser.add_argument("--once", action="store_true", help="Fetch once and exit")
    parser.add_argument("--fresh", action="store_true", help="Ignore the saved snapshot on startup")
    parser.add_argument("--no-state", action="store_true", help="Do not read or write snapshot state")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--bell", action="store_true", help="Ring the terminal bell on changes")
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--team", default=os.environ.get("SNUAICHAL_TEAM", DEFAULT_TEAM))
    parser.add_argument("-n", "--show", type=int, default=DEFAULT_SHOW, help="0 shows every team")
    parser.add_argument("--projected-cut", type=float, default=DEFAULT_PROJECTED_CUT)
    parser.add_argument(
        "--start",
        type=parse_schedule_time,
        default=COMPETITION_START,
        help="Preliminary start in ISO format; timezone-less values use KST",
    )
    parser.add_argument(
        "--deadline",
        type=parse_schedule_time,
        default=COMPETITION_DEADLINE,
        help="Submission deadline in ISO format; timezone-less values use KST",
    )
    parser.add_argument("--exclude-team", action="append", default=[])
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("outputs/leaderboard_watch/state.json"),
    )
    parser.add_argument(
        "--events-file",
        type=Path,
        default=Path("outputs/leaderboard_watch/events.jsonl"),
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.once and args.interval < 5:
        parser.error("--interval must be at least 5 seconds")
    if args.timeout <= 0 or args.max_backoff <= 0:
        parser.error("--timeout and --max-backoff must be positive")
    if args.show < 0:
        parser.error("--show must be zero or positive")
    if not 0 <= args.projected_cut <= 1:
        parser.error("--projected-cut must be between 0 and 1")
    if not args.team.strip():
        parser.error("--team must not be empty")
    if args.deadline <= args.start:
        parser.error("--deadline must be later than --start")


def main(argv: Sequence[str] | None = None) -> int:
    global COLOR_ENABLED

    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    COLOR_ENABLED = not args.no_color and "NO_COLOR" not in os.environ and sys.stdout.isatty()
    _enable_windows_ansi()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        credentials = load_credentials()
        previous = None if args.fresh or args.no_state else load_snapshot(
            args.state_file, competition=args.competition
        )
    except LeaderboardError as exc:
        parser.exit(2, f"error: {exc}\n")

    print(
        paint(
            f"\n  SNU AI Challenge 리더보드 감시 · {args.interval:g}초 간격 · "
            f"우리 팀 = {args.team} · 인증 = {credentials.source}",
            "bold",
            "cyan",
        ),
        flush=True,
    )

    failures = 0
    while True:
        try:
            current = fetch_leaderboard(
                credentials,
                competition=args.competition,
                timeout=args.timeout,
                excluded_teams=args.exclude_team,
            )
            failures = 0
        except KeyboardInterrupt:
            print(paint("\n  감시 종료.", "dim"))
            return 0
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, LeaderboardError) as exc:
            failures += 1
            if args.once:
                print(f"error: leaderboard fetch failed: {exc}", file=sys.stderr)
                return 1
            delay = min(args.max_backoff, max(args.interval, 5) * (2 ** min(failures - 1, 5)))
            delay *= random.uniform(0.9, 1.1)
            stamp = datetime.now(KST).strftime("%H:%M:%S")
            print(
                f"{paint('[' + stamp + ']', 'dim')} "
                f"{paint(f'조회 실패 ({failures}회): {exc}; {delay:.0f}초 후 재시도', 'red')}",
                flush=True,
            )
            time.sleep(delay)
            continue

        events = diff_events(previous, current) if previous is not None else []
        if previous is None:
            print(paint("  초기 스냅샷 취득", "dim"), flush=True)
            render_table(
                current,
                our_team=args.team,
                show=args.show,
                projected_cut=args.projected_cut,
                competition_start=args.start,
                competition_deadline=args.deadline,
            )
        elif events:
            stamp = datetime.now(KST).strftime("%H:%M:%S")
            print(f"\n{paint('[' + stamp + '] ◆ 갱신 감지', 'yellow', 'bold')} — {len(events)}건")
            for event in events:
                print("    " + format_event(event, our_team=args.team), flush=True)
            append_events(args.events_file, events)
            if args.bell:
                print("\a", end="", flush=True)
            render_table(
                current,
                our_team=args.team,
                show=args.show,
                projected_cut=args.projected_cut,
                competition_start=args.start,
                competition_deadline=args.deadline,
            )
        elif args.table or args.once:
            render_table(
                current,
                our_team=args.team,
                show=args.show,
                projected_cut=args.projected_cut,
                competition_start=args.start,
                competition_deadline=args.deadline,
            )
        else:
            stamp = datetime.now(KST).strftime("%H:%M:%S")
            countdown = competition_status_text(
                start=args.start,
                deadline=args.deadline,
                compact=True,
            )
            print(
                f"\r{paint('[' + stamp + '] 변동 없음 · ' + countdown, 'dim')}   ",
                end="",
                flush=True,
            )

        if not args.no_state:
            save_snapshot(args.state_file, competition=args.competition, rows=current)
        if args.once:
            return 0
        previous = current
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print(paint("\n  감시 종료.", "dim"))
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
