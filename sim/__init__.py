"""Constraint-aware VLA simulation pipeline (SO-101 + MuJoCo).

See sim/README.md for the bigger picture. Public surface:

    from sim import SafeCubeEnv, EnvConfig, SceneConfig, ExpertConfig
    from sim.expert import ScriptedExpert
    from sim.recorder import EpisodeRecorder
    from sim.evaluate import evaluate
"""

from .configs import EnvConfig, ExpertConfig, SceneConfig
from .env import SafeCubeEnv

__all__ = ["EnvConfig", "ExpertConfig", "SceneConfig", "SafeCubeEnv"]
