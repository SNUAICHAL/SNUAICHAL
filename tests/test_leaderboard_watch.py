import base64
import json
from datetime import timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from snuaichal.leaderboard_watch import (
    COMPETITION_DEADLINE,
    COMPETITION_START,
    DEFAULT_TEAM,
    Credentials,
    LeaderboardError,
    _parser,
    competition_progress,
    competition_status_text,
    diff_events,
    fetch_leaderboard,
    format_duration,
    load_credentials,
    load_snapshot,
    parse_submission_time,
    save_snapshot,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_load_credentials_uses_basic_auth_without_leaking_secret(tmp_path: Path) -> None:
    path = tmp_path / "kaggle.json"
    path.write_text(
        json.dumps({"username": "rice", "key": "super-secret"}), encoding="utf-8"
    )

    credentials = load_credentials(environ={}, config_path=path)

    expected = base64.b64encode(b"rice:super-secret").decode("ascii")
    assert credentials.authorization == f"Basic {expected}"
    assert "super-secret" not in repr(credentials)


def test_load_credentials_requires_username_with_classic_key(tmp_path: Path) -> None:
    path = tmp_path / "kaggle.json"
    path.write_text(json.dumps({"key": "secret"}), encoding="utf-8")

    with pytest.raises(LeaderboardError, match="KAGGLE_USERNAME"):
        load_credentials(environ={}, config_path=path)


def test_parse_submission_time_accepts_fractional_utc_timestamp() -> None:
    parsed = parse_submission_time("2026-07-12T23:14:25.6500000Z")

    assert parsed.isoformat().startswith("2026-07-12T23:14:25.650000")
    assert parsed.utcoffset().total_seconds() == 0


def test_fetch_leaderboard_paginates_deduplicates_and_normalizes_team() -> None:
    pages = {
        "": {
            "submissions": [
                {
                    "teamName": "first",
                    "score": "0.95",
                    "submissionDate": "2026-07-12T23:14:25Z",
                },
                {
                    "teamName": "exclude me",
                    "score": "0.94",
                    "submissionDate": "2026-07-12T22:14:25Z",
                },
            ],
            "nextPageToken": "next page",
        },
        "next page": {
            "submissions": [
                {
                    "teamName": "first",
                    "score": "0.93",
                    "submissionDate": "2026-07-11T23:14:25Z",
                },
                {
                    "teamName": DEFAULT_TEAM,
                    "score": "0.90",
                    "submissionDate": "2026-07-13T01:00:00Z",
                },
            ]
        },
    }
    seen_authorization = []

    def opener(request, timeout):
        assert timeout == 3
        seen_authorization.append(request.get_header("Authorization"))
        page = parse_qs(urlparse(request.full_url).query).get("pageToken", [""])[0]
        return FakeResponse(pages[page])

    rows = fetch_leaderboard(
        Credentials("Bearer hidden", "test"),
        timeout=3,
        excluded_teams=["exclude me"],
        opener=opener,
    )

    assert list(rows) == ["first", DEFAULT_TEAM]
    assert rows["first"]["rank"] == 1
    assert rows[DEFAULT_TEAM]["rank"] == 2
    assert rows[DEFAULT_TEAM]["updated_at"] == "2026-07-13 10:00:00"
    assert seen_authorization == ["Bearer hidden", "Bearer hidden"]


def test_fetch_leaderboard_rejects_empty_response() -> None:
    def opener(_request, timeout):
        assert timeout == 15
        return FakeResponse({"submissions": []})

    with pytest.raises(LeaderboardError, match="empty leaderboard"):
        fetch_leaderboard(Credentials("Bearer hidden", "test"), opener=opener)


def test_diff_events_reports_score_rank_and_tier_cut_changes() -> None:
    previous = {
        f"team-{index}": {
            "team": f"team-{index}",
            "rank": index,
            "score": 1 - index / 100,
            "updated_at": "old",
        }
        for index in range(1, 18)
    }
    current = {team: dict(row) for team, row in previous.items()}
    current["team-1"].update(score=0.995, updated_at="new")
    current["team-7"].update(score=0.94)
    current["team-8"].update(rank=7)
    current["team-7"].update(rank=8)

    events = diff_events(previous, current)

    assert any(event["kind"] == "score" and event["team"] == "team-1" for event in events)
    assert any(event["kind"] == "rank" and event["team"] == "team-8" for event in events)
    assert any(event["kind"] == "cut" and event["rank"] == 7 for event in events)


def test_snapshot_round_trip_and_competition_guard(tmp_path: Path) -> None:
    rows = {
        DEFAULT_TEAM: {
            "team": DEFAULT_TEAM,
            "rank": 1,
            "score": 0.9,
            "updated_at": "2026-07-13 10:00:00",
        }
    }
    path = tmp_path / "state.json"

    save_snapshot(path, competition="snuaichallenge", rows=rows)

    assert load_snapshot(path, competition="snuaichallenge") == rows
    assert load_snapshot(path, competition="different") is None
    assert "authorization" not in path.read_text(encoding="utf-8")


def test_cli_default_team_is_the_requested_team_name() -> None:
    args = _parser().parse_args([])

    assert args.team == "밥먹을돈으로3090사서거지됨"
    assert args.start == COMPETITION_START
    assert args.deadline == COMPETITION_DEADLINE


def test_competition_progress_before_during_and_after_preliminary() -> None:
    before = competition_progress(COMPETITION_START - timedelta(hours=2))
    midpoint = COMPETITION_START + (COMPETITION_DEADLINE - COMPETITION_START) / 2
    running = competition_progress(midpoint)
    finished = competition_progress(COMPETITION_DEADLINE + timedelta(seconds=1))

    assert before.phase == "upcoming"
    assert before.ratio == 0
    assert before.remaining == timedelta(hours=2)
    assert running.phase == "running"
    assert running.ratio == pytest.approx(0.5)
    assert running.elapsed == running.remaining
    assert finished.phase == "finished"
    assert finished.ratio == 1
    assert finished.remaining == timedelta(0)


def test_competition_status_contains_progress_and_remaining_time() -> None:
    at = COMPETITION_DEADLINE - timedelta(days=2, hours=3, minutes=4, seconds=5)

    status = competition_status_text(at=at)
    compact = competition_status_text(at=at, compact=True)

    assert "%" in status
    assert "남은 시간 2일 3시간 4분 5초" in status
    assert "마감까지 2일 3시간 4분 5초" in compact
    assert format_duration(timedelta(seconds=65)) == "1분 5초"
