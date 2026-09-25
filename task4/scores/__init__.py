"""Unknownness scores u(x): larger values = greater novelty.

Every function takes the *saved* outputs of a frozen model (logits / features),
so all scores for one model are computed on exactly the same examples.
"""
from scores.energy import energy_score
from scores.mahalanobis import MahalanobisScorer, mahalanobis_score
from scores.mls import mls_score
from scores.msp import msp_score
from scores.proser_dummy import proser_dummy_score, proser_margin_score
from scores.rpl_distance import rpl_score

__all__ = ["msp_score", "mls_score", "energy_score", "MahalanobisScorer", "mahalanobis_score",
           "proser_dummy_score", "proser_margin_score", "rpl_score"]

SCORE_LABELS = {
    "msp": "MSP",
    "mls": "MLS",
    "energy": "Energy",
    "mahalanobis": "Mahalanobis",
    "proser_dummy": "PROSER dummy (placeholder)",
    "proser_margin": "PROSER dummy margin (bias-calibrated)",
    "rpl": "RPL max-distance",
}
