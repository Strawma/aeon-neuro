"""Factories for unfitted label-free channel selectors."""

# These label-free selectors are shared by old preparation recipes and per-fold fitting.
UNSUPERVISED_SELECTORS = {
    "variance": "Highest variance",
    "clever_rank": "CLeVer rank",
    "clever_cluster": "CLeVer cluster",
    "clever_hybrid": "CLeVer hybrid",
}


def make_unsupervised_selector(method, keep, n_channels, seed=0):
    """Return an unfitted selector for one of the supported label-free methods."""
    from aeon_neuro.transformations.collection.channel_selection import (
        ChannelFilter,
        CLeVerCluster,
        CLeVerHybrid,
        CLeVerRank,
    )

    return {
        "variance": lambda: ChannelFilter(
            score_channel="variance", proportion=keep / n_channels
        ),
        "clever_rank": lambda: CLeVerRank(n_channels=keep),
        "clever_cluster": lambda: CLeVerCluster(n_channels=keep, random_state=seed),
        "clever_hybrid": lambda: CLeVerHybrid(n_channels=keep, random_state=seed),
    }[method]()
