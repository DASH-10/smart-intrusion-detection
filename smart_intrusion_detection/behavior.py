from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .config import BehaviorConfig
from .tracking import Track


@dataclass
class IntrusionEvent:
    track_id: int
    zone_id: str
    event: str  # "enter" or "dwell"
    timestamp: float
    dwell_time: float | None = None


class BehaviorAnalyzer:
    def __init__(self, cfg: BehaviorConfig, restricted_zone_ids: Sequence[str]):
        self.cfg = cfg
        self.restricted_zone_ids = set(restricted_zone_ids)
        self._state: Dict[Tuple[int, str], Dict[str, float]] = {}

    def process(self, tracks: Iterable[Track], now: float | None = None) -> List[IntrusionEvent]:
        """Compute intrusion/loitering events for the given tracks."""
        timestamp = now or time.time()
        events: List[IntrusionEvent] = []
        active_keys = set()

        for track in tracks:
            zone_ids = getattr(track, "zone_ids", [])
            for zone_id in zone_ids:
                active_keys.add((track.track_id, zone_id))
                if zone_id not in self.restricted_zone_ids:
                    continue

                key = (track.track_id, zone_id)
                state = self._state.get(key)
                if state is None:
                    self._state[key] = {"entered": timestamp, "last_alert": 0.0}
                    events.append(IntrusionEvent(track_id=track.track_id, zone_id=zone_id, event="enter", timestamp=timestamp))
                else:
                    dwell = timestamp - state["entered"]
                    if dwell >= self.cfg.dwell_time_seconds and (timestamp - state["last_alert"] >= self.cfg.cooldown_seconds):
                        state["last_alert"] = timestamp
                        events.append(
                            IntrusionEvent(
                                track_id=track.track_id,
                                zone_id=zone_id,
                                event="dwell",
                                dwell_time=dwell,
                                timestamp=timestamp,
                            )
                        )

        # Remove tracks that have left all zones.
        for key in list(self._state.keys()):
            if key not in active_keys:
                self._state.pop(key, None)

        return events
