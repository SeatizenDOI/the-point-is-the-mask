from enum import Enum

from ..ConfigParser import ConfigParser


class TrainingStep(Enum):
    COARSE = "coarse"
    REFINE = "refine"


def resume_from_training_step(cp: ConfigParser, ts: TrainingStep) -> str | None:
    """ Return the str of resume based on training step"""
    if ts == TrainingStep.COARSE:
        return cp.resume_coarse_training
    return cp.resume_refine_training
