from lmlab.multimodal import paired_bootstrap


def test_paired_bootstrap_is_reproducible():
    first = paired_bootstrap([0, 1, 0, 1], [1, 1, 0, 1], repeats=500, seed=7)
    second = paired_bootstrap([0, 1, 0, 1], [1, 1, 0, 1], repeats=500, seed=7)
    assert first == second
    assert first["mean_difference"] == 0.25

