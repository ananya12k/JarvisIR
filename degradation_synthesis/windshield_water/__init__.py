"""
Stateful windshield-water simulator (separate from the earlier temporal_dynamic mode).

Module A (wet appearance) is MUNIT, applied by the driver script.
Module B  droplet.py, simulator.py      persistent droplets (ids, motion, lifecycles)
Module C  interactions.py, trails.py,   growth, sliding, merging, trails, rivulets,
          film.py, splashes.py          film, splashes
Module D  rasterizer.py, renderer.py    state -> height / normals / offsets / alpha -> frame
config.py presets (light / steady / heavy / torrential) + validation; metrics.py measurements.

Windshield water only: atmospheric rain streaks between camera and scene are a
separate effect and deliberately not part of this droplet system.
"""

from .config import SCENARIOS, SimConfig, apply_overrides, preset  # noqa: F401
from .droplet import Droplet  # noqa: F401
from .metrics import MetricsRecorder, mask_change  # noqa: F401
from .rasterizer import WaterMaps  # noqa: F401
from .simulator import WindshieldWaterSimulator  # noqa: F401
