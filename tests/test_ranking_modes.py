import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from artist_elo_ranker import (
    ActivePool,
    ELOSystem,
    POOL_ACTION_CALIBRATE_SOLO,
    POOL_ACTION_EXPAND_TO_200,
    POOL_ACTION_REFILL_FROM_150,
    POOL_ACTION_TRIM_FROM_200,
    POOL_ACTION_TRIM_TO_150,
    RANKING_MODE_FAST_ROTATION,
    RANKING_MODE_NEWCOMERS,
    RANKING_MODE_TOP,
)


class RankingModeTests(unittest.TestCase):
    def make_pool(
        self,
        all_artists,
        active_artists=None,
        ratings=None,
        comparisons=None,
        pool_size=150,
    ):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        elo = ELOSystem(
            ratings=ratings or {},
            artist_comparisons=comparisons or {},
        )
        pool_file = Path(temp_dir.name) / "active_pool.json"
        pool = ActivePool(
            all_artists,
            elo,
            pool_size=min(pool_size, len(all_artists)),
            pool_file=pool_file,
        )
        pool.pool = list(active_artists or all_artists[:pool_size])
        pool.pool_size = pool_size
        pool.save()
        return pool, pool_file

    @staticmethod
    def risk_fixture(pool_count):
        regular = [f"regular_{index}" for index in range(pool_count - 3)]
        active = ["risk_1", "risk_2", "high"] + regular
        ratings = {artist: 1500 for artist in active}
        ratings.update({"risk_1": 1200, "risk_2": 1300, "high": 1800})
        comparisons = {artist: 8 for artist in active}
        return active, ratings, comparisons

    def test_mode_is_persisted_with_the_pool(self):
        pool, pool_file = self.make_pool(["a", "b", "c"], pool_size=3)

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

    def test_new_mode_below_200_selects_two_outside_artists_as_solos(self):
        active = [f"active_{index}" for index in range(150)]
        outside = ["outside_1", "outside_2", "outside_3"]
        pool, _ = self.make_pool(active + outside, active_artists=active)
        pool.set_ranking_mode(RANKING_MODE_NEWCOMERS)

        with patch("artist_elo_ranker.random.uniform", return_value=0.0):
            artists_a, artists_b, action = pool.select_comparison_pair()

        self.assertEqual(action, POOL_ACTION_EXPAND_TO_200)
        self.assertEqual(artists_a, ["outside_1"])
        self.assertEqual(artists_b, ["outside_2"])

    def test_new_mode_at_200_selects_at_risk_artists_as_solos(self):
        active, ratings, comparisons = self.risk_fixture(200)
        pool, _ = self.make_pool(
            active,
            active_artists=active,
            ratings=ratings,
            comparisons=comparisons,
        )
        pool.set_ranking_mode(RANKING_MODE_NEWCOMERS)

        with patch("artist_elo_ranker.random.uniform", return_value=0.0):
            artists_a, artists_b, action = pool.select_comparison_pair()

        self.assertEqual(action, POOL_ACTION_TRIM_FROM_200)
        self.assertEqual(artists_a, ["risk_1"])
        self.assertEqual(artists_b, ["risk_2"])

    def test_replacement_mode_above_150_selects_at_risk_solos(self):
        active, ratings, comparisons = self.risk_fixture(151)
        pool, _ = self.make_pool(
            active,
            active_artists=active,
            ratings=ratings,
            comparisons=comparisons,
        )
        pool.set_ranking_mode(RANKING_MODE_FAST_ROTATION)

        with patch("artist_elo_ranker.random.uniform", return_value=0.0):
            artists_a, artists_b, action = pool.select_comparison_pair()

        self.assertEqual(action, POOL_ACTION_TRIM_TO_150)
        self.assertEqual(artists_a, ["risk_1"])
        self.assertEqual(artists_b, ["risk_2"])

    def test_replacement_mode_at_150_selects_outside_solos(self):
        active = [f"active_{index}" for index in range(150)]
        outside = ["outside_1", "outside_2"]
        pool, _ = self.make_pool(active + outside, active_artists=active)
        pool.set_ranking_mode(RANKING_MODE_FAST_ROTATION)

        with patch("artist_elo_ranker.random.uniform", return_value=0.0):
            artists_a, artists_b, action = pool.select_comparison_pair()

        self.assertEqual(action, POOL_ACTION_REFILL_FROM_150)
        self.assertEqual(artists_a, ["outside_1"])
        self.assertEqual(artists_b, ["outside_2"])

    def test_outside_solo_result_adds_both_compared_artists(self):
        active = [f"active_{index}" for index in range(150)]
        outside = ["outside_1", "outside_2"]
        pool, _ = self.make_pool(active + outside, active_artists=active)
        pool.elo_system.ratings.update({"outside_1": 1516, "outside_2": 1484})
        pool.elo_system.artist_comparisons.update({"outside_1": 1, "outside_2": 1})

        _, rotated_in = pool.process_result(
            ["outside_1"],
            ["outside_2"],
            POOL_ACTION_EXPAND_TO_200,
        )

        self.assertEqual(len(pool.pool), 152)
        self.assertCountEqual(
            [artist for artist, _, _ in rotated_in],
            outside,
        )
        self.assertTrue(all(not is_returning for _, _, is_returning in rotated_in))

    def test_at_risk_result_removes_the_losing_artist_at_200(self):
        active, ratings, comparisons = self.risk_fixture(200)
        pool, _ = self.make_pool(
            active,
            active_artists=active,
            ratings=ratings,
            comparisons=comparisons,
        )

        rotated_out, _ = pool.process_result(
            ["risk_1"],
            ["risk_2"],
            POOL_ACTION_TRIM_FROM_200,
        )

        self.assertEqual(len(pool.pool), 199)
        self.assertEqual(rotated_out, [("risk_2", 1300)])
        self.assertNotIn("risk_2", pool.pool)

    def test_at_risk_result_stops_replacement_pool_at_150(self):
        active, ratings, comparisons = self.risk_fixture(151)
        pool, _ = self.make_pool(
            active,
            active_artists=active,
            ratings=ratings,
            comparisons=comparisons,
        )

        pool.process_result(
            ["risk_1"],
            ["risk_2"],
            POOL_ACTION_TRIM_TO_150,
        )
        rotated_out, _ = pool.process_result(
            ["risk_1"],
            ["high"],
            POOL_ACTION_TRIM_TO_150,
        )

        self.assertEqual(len(pool.pool), 150)
        self.assertEqual(rotated_out, [])

    def test_no_at_risk_candidates_uses_solo_calibration(self):
        active = [f"active_{index}" for index in range(200)]
        pool, _ = self.make_pool(active, active_artists=active)
        pool.set_ranking_mode(RANKING_MODE_NEWCOMERS)

        with patch("artist_elo_ranker.random.uniform", return_value=0.0):
            artists_a, artists_b, action = pool.select_comparison_pair()

        self.assertEqual(action, POOL_ACTION_CALIBRATE_SOLO)
        self.assertEqual(len(artists_a), 1)
        self.assertEqual(len(artists_b), 1)
        self.assertNotEqual(artists_a, artists_b)

    def test_top_mode_focuses_on_top_confident_artists(self):
        artists = ["top", "second", "middle", "low", "unrated"]
        pool, _ = self.make_pool(
            artists,
            active_artists=artists,
            ratings={
                "top": 1700,
                "second": 1600,
                "middle": 1500,
                "low": 1300,
            },
            comparisons={artist: 8 for artist in artists[:-1]},
            pool_size=5,
        )
        pool.set_ranking_mode(RANKING_MODE_TOP)

        focused = pool._get_focus_candidates(artists)

        self.assertEqual(focused, ["top", "second"])
        self.assertNotIn("unrated", focused)

    def test_revert_rotation_restores_original_membership(self):
        pool, _ = self.make_pool(["a", "b", "c", "d"], pool_size=3)
        pool.pool = ["a", "b", "c"]
        pool.save()
        pool.pool.remove("a")
        pool.pool.append("d")

        pool.revert_rotation(rotated_out=["a"], rotated_in=["d"])

        self.assertCountEqual(pool.pool, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
