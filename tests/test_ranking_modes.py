import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from artist_elo_ranker import (
    ActivePool,
    ELOSystem,
    RANKING_MODE_FAST_ROTATION,
    RANKING_MODE_NEWCOMERS,
    RANKING_MODE_TOP,
)


class RankingModeTests(unittest.TestCase):
    def make_pool(self, artists, ratings=None, comparisons=None):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        elo = ELOSystem(
            ratings=ratings or {},
            artist_comparisons=comparisons or {},
        )
        pool_file = Path(temp_dir.name) / "active_pool.json"
        pool = ActivePool(
            artists,
            elo,
            pool_size=len(artists),
            pool_file=pool_file,
        )
        pool.pool = list(artists)
        pool.save()
        return pool, pool_file

    def test_mode_is_persisted_with_the_pool(self):
        pool, pool_file = self.make_pool(["a", "b", "c"])

        pool.set_ranking_mode(RANKING_MODE_NEWCOMERS)

        saved = json.loads(pool_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["ranking_mode"], RANKING_MODE_NEWCOMERS)
        reloaded = ActivePool(
            ["a", "b", "c"],
            pool.elo_system,
            pool_size=3,
            pool_file=pool_file,
        )
        self.assertEqual(reloaded.get_ranking_mode(), RANKING_MODE_NEWCOMERS)

    def test_newcomer_mode_focuses_on_under_five_comparisons(self):
        pool, _ = self.make_pool(
            ["established", "new"],
            comparisons={"established": 10, "new": 0},
        )
        pool.set_ranking_mode(RANKING_MODE_NEWCOMERS)

        with patch("artist_elo_ranker.random.random", return_value=0.0), patch(
            "artist_elo_ranker.random.uniform", return_value=0.0
        ):
            self.assertEqual(pool.select_artist(), "new")

    def test_top_mode_focuses_on_top_confident_artists(self):
        artists = ["top", "second", "middle", "low", "unrated"]
        pool, _ = self.make_pool(
            artists,
            ratings={
                "top": 1700,
                "second": 1600,
                "middle": 1500,
                "low": 1300,
            },
            comparisons={artist: 8 for artist in artists[:-1]},
        )
        pool.set_ranking_mode(RANKING_MODE_TOP)

        focused = pool._get_focus_candidates(artists)

        self.assertEqual(focused, ["top", "second"])
        self.assertNotIn("unrated", focused)

    def test_fast_rotation_focuses_on_confident_below_average_artists(self):
        artists = ["high", "average", "low", "new_low"]
        pool, _ = self.make_pool(
            artists,
            ratings={
                "high": 1700,
                "average": 1500,
                "low": 1300,
                "new_low": 1200,
            },
            comparisons={"high": 8, "average": 8, "low": 8, "new_low": 2},
        )
        pool.set_ranking_mode(RANKING_MODE_FAST_ROTATION)

        focused = pool._get_focus_candidates(artists)

        self.assertEqual(focused, ["low"])
        self.assertNotIn("new_low", focused)

    def test_revert_rotation_restores_original_membership(self):
        pool, _ = self.make_pool(["a", "b", "c", "d"])
        pool.pool = ["a", "b", "c"]
        pool.save()
        pool.pool.remove("a")
        pool.pool.append("d")

        pool.revert_rotation(rotated_out=["a"], rotated_in=["d"])

        self.assertCountEqual(pool.pool, ["a", "b", "c"])

    def test_pool_removal_protects_artists_under_five_comparisons(self):
        artists = ["new_low", "confident_low", "high", "outside"]
        pool, _ = self.make_pool(
            artists,
            ratings={
                "new_low": 1100,
                "confident_low": 1300,
                "high": 1800,
                "outside": 1500,
            },
            comparisons={
                "new_low": 2,
                "confident_low": 8,
                "high": 8,
                "outside": 8,
            },
        )
        pool.pool = ["new_low", "confident_low", "high"]
        pool.pool_size = 3
        pool.set_ranking_mode(RANKING_MODE_FAST_ROTATION)

        with patch(
            "artist_elo_ranker.random.random", side_effect=[0.0, 1.0]
        ), patch("artist_elo_ranker.random.uniform", return_value=0.0):
            rotated_out, _ = pool.process_result(["high"], ["confident_low"])

        self.assertEqual(rotated_out[0][0], "confident_low")
        self.assertIn("new_low", pool.pool)


if __name__ == "__main__":
    unittest.main()
