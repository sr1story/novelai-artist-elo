import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from artist_elo_ranker import (
    ActivePool,
    ArtistELORanker,
    ArtistTagManager,
    CANDIDATE_RULE_DARK_HORSE,
    CANDIDATE_RULE_FAMILIAR,
    CANDIDATE_RULE_NEW,
    CANDIDATE_RULE_PROVEN,
    ComparisonHistory,
    DeathmatchPool,
    ELOSystem,
    GenerationSettings,
    HallOfFamePool,
    PromptPresetStore,
    POOL_ACTION_CALIBRATE_SOLO,
    POOL_ACTION_EXPAND_TO_200,
    POOL_ACTION_REFILL_FROM_150,
    POOL_ACTION_TRIM_FROM_200,
    POOL_ACTION_TRIM_TO_150,
    POOL_ACTION_TEMPORARY,
    RANKING_MODE_BOTTOM,
    RANKING_MODE_FAST_ROTATION,
    RANKING_MODE_NEWCOMERS,
    RANKING_MODE_TOP,
    COMPARISON_MODE_NORMAL,
    COMPARISON_MODE_SOLO,
    COMPARISON_MODE_WEIGHTED,
    generate_comparison_pair,
    generate_deathmatch_pair,
    generate_image,
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
        pool.set_candidate_rule(CANDIDATE_RULE_DARK_HORSE)

        saved = json.loads(pool_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["ranking_mode"], RANKING_MODE_NEWCOMERS)
        self.assertEqual(saved["candidate_rule"], CANDIDATE_RULE_DARK_HORSE)
        reloaded = ActivePool(
            ["a", "b", "c"],
            pool.elo_system,
            pool_size=3,
            pool_file=pool_file,
        )
        self.assertEqual(reloaded.get_ranking_mode(), RANKING_MODE_NEWCOMERS)
        self.assertEqual(
            reloaded.get_candidate_rule(),
            CANDIDATE_RULE_DARK_HORSE,
        )

    def test_candidate_rules_use_comparisons_and_relative_elo(self):
        artists = ["new", "dark", "exploring", "familiar", "proven"]
        pool, _ = self.make_pool(
            artists,
            active_artists=artists,
            ratings={
                "new": 1480,
                "dark": 1600,
                "exploring": 1300,
                "familiar": 1450,
                "proven": 1800,
            },
            comparisons={
                "new": 2,
                "dark": 6,
                "exploring": 6,
                "familiar": 12,
                "proven": 12,
            },
            pool_size=5,
        )

        self.assertEqual(pool.get_artist_candidate_label("new"), "새로운")
        self.assertEqual(pool.get_artist_candidate_label("dark"), "다크호스")
        self.assertEqual(pool.get_artist_candidate_label("exploring"), "탐색 중")
        self.assertEqual(pool.get_artist_candidate_label("familiar"), "친숙한")
        self.assertEqual(pool.get_artist_candidate_label("proven"), "검증된 강자")

        expected = {
            CANDIDATE_RULE_NEW: ["new"],
            CANDIDATE_RULE_DARK_HORSE: ["dark"],
            CANDIDATE_RULE_FAMILIAR: ["familiar", "proven"],
            CANDIDATE_RULE_PROVEN: ["proven"],
        }
        for rule, candidates in expected.items():
            pool.set_candidate_rule(rule)
            self.assertEqual(
                pool._get_candidate_rule_candidates(artists),
                candidates,
            )

    def test_candidate_rule_focuses_standard_selection(self):
        artists = ["dark", "other_a", "other_b"]
        pool, _ = self.make_pool(
            artists,
            active_artists=artists,
            ratings={"dark": 1700, "other_a": 1400, "other_b": 1400},
            comparisons={"dark": 6, "other_a": 6, "other_b": 12},
            pool_size=3,
        )
        pool.set_candidate_rule(CANDIDATE_RULE_DARK_HORSE)

        with (
            patch("artist_elo_ranker.random.random", return_value=0.0),
            patch("artist_elo_ranker.random.uniform", return_value=0.0),
        ):
            selected = pool._select_from_candidates(artists)

        self.assertEqual(selected, "dark")

    def test_pool_out_count_only_includes_evaluated_inactive_artists(self):
        pool, _ = self.make_pool(
            ["active_a", "active_b", "removed", "never_seen"],
            active_artists=["active_a", "active_b"],
            ratings={"active_a": 1510, "active_b": 1490, "removed": 1400},
            comparisons={"active_a": 4, "active_b": 4, "removed": 7},
            pool_size=2,
        )

        stats = pool.get_pool_stats()

        self.assertEqual(stats["out_count"], 1)

    def test_artist_text_extraction_removes_non_artists_and_duplicates(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        tags_file = Path(temp_dir.name) / "artists.txt"
        tags_file.write_text(
            "alpha artist\n723_nanahumi\nlococo:p\nbeta (test)\n",
            encoding="utf-8",
        )
        manager = ArtistTagManager(
            tags_file,
            ELOSystem(),
            Path(temp_dir.name) / "temporary_pool.json",
        )

        artists, ignored_count = manager.extract_artists_from_text(
            "quality, artist: alpha_artist, {artist: lococo:p:1.2}, "
            "- 723 nanahumi, alpha artist, bad anatomy"
        )

        self.assertEqual(
            artists,
            ["alpha artist", "lococo:p", "723_nanahumi"],
        )
        self.assertEqual(ignored_count, 2)

    def test_artist_text_extraction_accepts_copied_statistics_tables(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        tags_file = Path(temp_dir.name) / "artists.txt"
        tags_file.write_text(
            "tenjin hidetaka\n"
            "sciamano240\n"
            "sousou (sousouworks)\n"
            "danart14020\n"
            "dandon fuga\n",
            encoding="utf-8",
        )
        manager = ArtistTagManager(
            tags_file,
            ELOSystem(),
            Path(temp_dir.name) / "temporary_pool.json",
        )
        copied_table = (
            "Name\tCosine\tJaccard\tOverlap\tFrequency\n"
            "? tenjin_hidetaka 533\t10.85%\t1.40%\t84.02%\t1.40%\n"
            "? sciamano240 1.0k\t7.17%\t1.26%\t40.18%\t1.28%\n"
            "? sousou_(sousouworks) 768 2.97% 0.45% 19.16% 0.46%\n"
            "? sakimichan 1.0k 1.90% 0.33% 10.67% 0.34%\n"
            "? danart14020 70 4.28% 0.20% 91.39% 0.20%\n"
            "? dandon_fuga 1.4k"
        )

        artists, ignored_count = manager.extract_artists_from_text(copied_table)

        self.assertEqual(
            artists,
            [
                "tenjin hidetaka",
                "sciamano240",
                "sousou (sousouworks)",
                "danart14020",
                "dandon fuga",
            ],
        )
        self.assertEqual(ignored_count, 1)

    def test_temporary_pool_persists_and_can_be_stopped_without_clearing(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        tags_file = Path(temp_dir.name) / "artists.txt"
        tags_file.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        temporary_pool_file = Path(temp_dir.name) / "temporary_pool.json"

        manager = ArtistTagManager(
            tags_file,
            ELOSystem(),
            temporary_pool_file,
        )
        manager.activate_temporary_pool(["alpha", "beta", "gamma"])

        reloaded = ArtistTagManager(
            tags_file,
            ELOSystem(),
            temporary_pool_file,
        )
        self.assertTrue(reloaded.temporary_pool_enabled)
        self.assertEqual(reloaded.temporary_pool, ["alpha", "beta", "gamma"])

        reloaded.deactivate_temporary_pool()
        stopped = ArtistTagManager(
            tags_file,
            ELOSystem(),
            temporary_pool_file,
        )
        self.assertFalse(stopped.temporary_pool_enabled)
        self.assertEqual(stopped.temporary_pool, ["alpha", "beta", "gamma"])

    def test_temporary_comparison_uses_normal_combinations_and_joins_main_pool(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        artists = ["active_a", "active_b", "temp_a", "temp_b", "temp_c"]
        tags_file = Path(temp_dir.name) / "artists.txt"
        tags_file.write_text("\n".join(artists) + "\n", encoding="utf-8")
        elo = ELOSystem()
        manager = ArtistTagManager(
            tags_file,
            elo,
            Path(temp_dir.name) / "temporary_pool.json",
        )
        manager.active_pool = ActivePool(
            artists,
            elo,
            pool_size=2,
            pool_file=Path(temp_dir.name) / "active_pool.json",
        )
        manager.active_pool.pool = ["active_a", "active_b"]
        manager.active_pool.save()
        manager.activate_temporary_pool(["temp_a", "temp_b", "temp_c"])
        artists_a, artists_b, action = manager.get_comparison_pair()
        rotated_out, rotated_in = manager.process_result(
            artists_a,
            artists_b,
            action,
        )

        self.assertEqual(action, POOL_ACTION_TEMPORARY)
        self.assertGreaterEqual(len(artists_a), 1)
        self.assertLessEqual(len(artists_a), 3)
        self.assertGreaterEqual(len(artists_b), 1)
        self.assertLessEqual(len(artists_b), 3)
        self.assertFalse(set(artists_a) & set(artists_b))
        used_temporary = (
            set(artists_a + artists_b) & {"temp_a", "temp_b", "temp_c"}
        )
        self.assertTrue(set(artists_a) & used_temporary)
        self.assertTrue(set(artists_b) & used_temporary)
        self.assertEqual(rotated_out, [])
        self.assertEqual(
            {artist for artist, _, _ in rotated_in},
            used_temporary,
        )
        self.assertTrue(used_temporary <= set(manager.active_pool.pool))
        self.assertFalse(used_temporary & set(manager.temporary_pool))

    def test_undo_first_temporary_vote_removes_new_rating_entries(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        artists = ["active_a", "active_b", "temp_a", "temp_b"]
        tags_file = Path(temp_dir.name) / "artists.txt"
        tags_file.write_text("\n".join(artists) + "\n", encoding="utf-8")
        elo = ELOSystem()
        manager = ArtistTagManager(
            tags_file,
            elo,
            Path(temp_dir.name) / "temporary_pool.json",
        )
        manager.active_pool = ActivePool(
            artists,
            elo,
            pool_size=2,
            pool_file=Path(temp_dir.name) / "active_pool.json",
        )
        manager.active_pool.pool = ["active_a", "active_b"]
        manager.active_pool.save()

        ranker = ArtistELORanker.__new__(ArtistELORanker)
        ranker.elo_system = elo
        ranker.artist_manager = manager
        ranker.history = ComparisonHistory(
            Path(temp_dir.name) / "comparison_history.json"
        )
        ranker.current_image_a = None
        ranker.current_image_b = None
        ranker.current_artists_a = ["temp_a"]
        ranker.current_artists_b = ["temp_b"]
        ranker.current_pool_action = POOL_ACTION_TEMPORARY
        ranker.current_generation_settings = GenerationSettings()
        ranker.current_quality_toggle = True
        ranker.current_uc_preset = 0
        ranker.current_pair_seed = 123
        ranker.rotation_log = []
        ranker.last_undo_state = None
        ranker.selection_made = False

        with patch.object(ELOSystem, "save"):
            ranker.pick_winner("A")
            self.assertIn("temp_a", elo.ratings)
            self.assertIn("temp_b", elo.ratings)
            ranker.undo_last_selection()

        self.assertNotIn("temp_a", elo.ratings)
        self.assertNotIn("temp_b", elo.ratings)
        self.assertNotIn("temp_a", elo.artist_comparisons)
        self.assertNotIn("temp_b", elo.artist_comparisons)
        self.assertEqual(manager.active_pool.pool, ["active_a", "active_b"])
        self.assertEqual(ranker.history.records, [])

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

    def test_replacement_selects_lowest_elo_regardless_of_comparison_count(self):
        active = [f"artist_{index}" for index in range(151)]
        ratings = {artist: 1500 for artist in active}
        ratings.update({"artist_0": 900, "artist_1": 1000, "artist_2": 1100})
        comparisons = {artist: 20 for artist in active}
        comparisons.update({"artist_0": 0, "artist_1": 1})
        pool, _ = self.make_pool(
            active,
            active_artists=active,
            ratings=ratings,
            comparisons=comparisons,
        )
        pool.set_ranking_mode(RANKING_MODE_FAST_ROTATION)
        pool.set_candidate_rule(CANDIDATE_RULE_DARK_HORSE)

        artists_a, artists_b, action = pool.select_comparison_pair()

        self.assertEqual(action, POOL_ACTION_TRIM_TO_150)
        self.assertCountEqual(artists_a + artists_b, ["artist_0", "artist_1"])

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

    def test_top_and_bottom_views_focus_opposite_elo_bands(self):
        artists = [f"artist_{index}" for index in range(10)]
        ratings = {
            artist: 1000 + index * 100
            for index, artist in enumerate(artists)
        }
        comparisons = {artist: 5 for artist in artists}
        pool, _ = self.make_pool(
            artists,
            active_artists=artists,
            ratings=ratings,
            comparisons=comparisons,
            pool_size=10,
        )

        pool.set_ranking_mode(RANKING_MODE_TOP)
        self.assertEqual(
            set(pool._get_focus_candidates(artists)),
            {"artist_7", "artist_8", "artist_9"},
        )

        pool.set_ranking_mode(RANKING_MODE_BOTTOM)
        self.assertEqual(
            set(pool._get_focus_candidates(artists)),
            {"artist_0", "artist_1", "artist_2"},
        )

    def test_revert_rotation_restores_original_membership(self):
        pool, _ = self.make_pool(["a", "b", "c", "d"], pool_size=3)
        pool.pool = ["a", "b", "c"]
        pool.save()
        pool.pool.remove("a")
        pool.pool.append("d")

        pool.revert_rotation(rotated_out=["a"], rotated_in=["d"])

        self.assertCountEqual(pool.pool, ["a", "b", "c"])

    def test_hall_of_fame_resets_hof_elo_and_preserves_main_elo(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        artists = ["a", "b", "c", "d"]
        main_elo = ELOSystem(
            ratings={"a": 1812},
            artist_comparisons={"a": 12},
        )
        main_pool = ActivePool(
            artists,
            main_elo,
            pool_size=4,
            pool_file=Path(temp_dir.name) / "active_pool.json",
        )
        hof_elo = ELOSystem(
            ratings={"a": 1930},
            artist_comparisons={"a": 9},
        )
        hall = HallOfFamePool(
            artists,
            hof_elo,
            main_pool,
            Path(temp_dir.name) / "hall.json",
        )

        self.assertEqual(hall.induct(["a"]), ["a"])
        self.assertNotIn("a", main_pool.pool)
        self.assertEqual(hof_elo.get_rating("a"), 1500)
        self.assertEqual(hof_elo.get_artist_comparison_count("a"), 0)

        hof_elo.ratings["a"] = 1725
        self.assertEqual(hall.return_to_main(["a"]), ["a"])
        self.assertIn("a", main_pool.pool)
        self.assertEqual(main_elo.get_rating("a"), 1812)

        hall.induct(["a"])
        self.assertEqual(hof_elo.get_rating("a"), 1500)
        self.assertEqual(hof_elo.get_artist_comparison_count("a"), 0)

    def test_hall_modes_are_disjoint_and_weighted_mode_is_balanced(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        artists = [f"artist_{index}" for index in range(20)]
        main_elo = ELOSystem()
        main_pool = ActivePool(
            artists,
            main_elo,
            pool_size=20,
            pool_file=Path(temp_dir.name) / "active_pool.json",
        )
        hof_elo = ELOSystem(
            ratings={artist: 1400 + index * 15 for index, artist in enumerate(artists)}
        )
        hall = HallOfFamePool(
            artists,
            hof_elo,
            main_pool,
            Path(temp_dir.name) / "hall.json",
        )
        hall.induct(artists)

        solo_a, solo_b = hall.select_pair(COMPARISON_MODE_SOLO)
        self.assertEqual((len(solo_a), len(solo_b)), (1, 1))
        self.assertFalse(set(solo_a) & set(solo_b))

        normal_a, normal_b = hall.select_pair(COMPARISON_MODE_NORMAL)
        self.assertIn(len(normal_a), {1, 2, 3})
        self.assertIn(len(normal_b), {1, 2, 3})
        self.assertFalse(set(normal_a) & set(normal_b))

        artists_a, artists_b, weights_a, weights_b = (
            hall.select_weighted_pair(10)
        )
        self.assertEqual(len(artists_a), 10)
        self.assertEqual(len(artists_b), 10)
        self.assertFalse(set(artists_a) & set(artists_b))
        self.assertAlmostEqual(sum(weights_a), sum(weights_b))
        self.assertTrue(all(0.5 <= weight <= 2.0 for weight in weights_a + weights_b))

    def test_image_star_round_trip_restores_main_rating(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        artists = ["a", "b", "c"]
        main_elo = ELOSystem(ratings={"a": 1777, "b": 1666})
        main_pool = ActivePool(
            artists,
            main_elo,
            pool_size=3,
            pool_file=Path(temp_dir.name) / "active_pool.json",
        )
        hof_elo = ELOSystem()
        hall = HallOfFamePool(
            artists,
            hof_elo,
            main_pool,
            Path(temp_dir.name) / "hall.json",
        )
        ranker = ArtistELORanker.__new__(ArtistELORanker)
        ranker.elo_system = main_elo
        ranker.hall_elo_system = hof_elo
        ranker.hall_pool = hall
        ranker.artist_manager = Mock(
            active_pool=main_pool,
            temporary_pool=[],
            temporary_pool_enabled=False,
        )
        ranker.artist_manager.save_temporary_pool = Mock()
        ranker.current_artists_a = ["a", "b"]
        ranker.current_artists_b = ["c"]
        ranker.current_pool_context = "main"
        ranker.current_side_actions = {"A": None, "B": None}
        ranker.current_image_a = None
        ranker.current_image_b = None
        ranker.selection_made = False

        with patch.object(ELOSystem, "save"):
            _, changed = ranker.apply_star_action("A")
            self.assertTrue(changed)
            self.assertCountEqual(hall.artists, ["a", "b"])
            self.assertEqual(hof_elo.get_rating("a"), 1500)
            self.assertTrue(ranker.selection_made)

            _, changed = ranker.apply_star_action("A")
            self.assertTrue(changed)

        self.assertNotIn("a", hall.artists)
        self.assertIn("a", main_pool.pool)
        self.assertEqual(main_elo.get_rating("a"), 1777)
        self.assertFalse(ranker.selection_made)

    def test_image_heart_queues_then_independent_up_down_resolves(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        artists = ["a", "b", "c", "d"]
        main_elo = ELOSystem(ratings={"a": 1750, "b": 1250})
        main_pool = ActivePool(
            artists,
            main_elo,
            pool_size=4,
            pool_file=Path(temp_dir.name) / "active_pool.json",
        )
        deathmatch = DeathmatchPool(
            artists,
            main_pool,
            Path(temp_dir.name) / "deathmatch.json",
        )
        ranker = ArtistELORanker.__new__(ArtistELORanker)
        ranker.elo_system = main_elo
        ranker.deathmatch_pool = deathmatch
        ranker.artist_manager = Mock(
            active_pool=main_pool,
            temporary_pool=[],
            temporary_pool_enabled=False,
        )
        ranker.artist_manager.save_temporary_pool = Mock()
        ranker.current_artists_a = ["a", "b"]
        ranker.current_artists_b = ["c"]
        ranker.current_pool_context = "main"
        ranker.current_side_actions = {"A": None, "B": None}
        ranker.current_image_a = None
        ranker.current_image_b = None
        ranker.selection_made = False

        _, changed = ranker.apply_broken_heart("A")
        self.assertTrue(changed)
        self.assertCountEqual(deathmatch.artists, ["a", "b"])
        self.assertEqual(ranker.current_side_actions["A"], "deathmatch")
        self.assertEqual(main_elo.get_rating("a"), 1750)

        ranker.current_pool_context = "deathmatch"
        ranker.current_artists_a = ["a"]
        ranker.current_artists_b = ["b"]
        ranker.current_side_actions = {"A": None, "B": None}
        ranker.selection_made = False
        _, changed = ranker.apply_deathmatch_decision("A", keep=True)
        self.assertTrue(changed)
        self.assertIn("a", main_pool.pool)
        self.assertFalse(ranker.selection_made)

        _, changed = ranker.apply_deathmatch_decision("B", keep=False)
        self.assertTrue(changed)
        self.assertNotIn("b", main_pool.pool)
        self.assertTrue(ranker.selection_made)

    def test_weighted_elo_is_zero_sum_and_scales_artist_influence(self):
        winners = ["winner_high", "winner_low", "winner_mid"]
        losers = ["loser_a", "loser_b", "loser_c"]
        system = ELOSystem(
            ratings={artist: 1500 for artist in winners + losers}
        )
        before_total = sum(system.ratings.values())

        system.update_weighted_ratings(
            winners,
            losers,
            [2.0, 0.5, 1.0],
            [1.0, 1.0, 1.0],
        )

        after_total = sum(system.ratings.values())
        self.assertAlmostEqual(before_total, after_total, places=6)
        self.assertGreater(
            system.get_rating("winner_high") - 1500,
            system.get_rating("winner_low") - 1500,
        )
        self.assertEqual(
            system.mode_comparisons["winner_high"][COMPARISON_MODE_WEIGHTED],
            1,
        )
        self.assertEqual(system.weighted_exposure["winner_high"], 2.0)

    def test_weighted_artist_prompt_uses_novelai_syntax(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        tags_file = Path(temp_dir.name) / "artists.txt"
        tags_file.write_text("alpha\nbeta\n", encoding="utf-8")
        manager = ArtistTagManager(
            tags_file,
            ELOSystem(),
            Path(temp_dir.name) / "temporary_pool.json",
        )

        prompt = manager.format_artist_tags(["alpha", "beta"], [1.3, 0.8])

        self.assertEqual(
            prompt,
            "1.3::artist: alpha::, 0.8::artist: beta::",
        )

    def test_broken_heart_exclusion_survives_refill(self):
        pool, _ = self.make_pool(
            ["a", "b", "c", "replacement"],
            active_artists=["a", "b", "c"],
            pool_size=3,
        )

        pool.remove_artists(["a"], permanent=True)
        pool._refill_pool()

        self.assertNotIn("a", pool.pool)
        self.assertIn("replacement", pool.pool)
        self.assertIn("a", pool.manual_excluded)

    def test_deathmatch_up_returns_and_down_confirms_pool_out(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        artists = ["a", "b", "c", "replacement"]
        elo = ELOSystem(
            ratings={"a": 1725, "b": 1280},
            artist_comparisons={"a": 8, "b": 8},
        )
        main_pool = ActivePool(
            artists,
            elo,
            pool_size=3,
            pool_file=Path(temp_dir.name) / "active_pool.json",
        )
        main_pool.pool = ["a", "b", "c"]
        main_pool.save()
        deathmatch_file = Path(temp_dir.name) / "deathmatch.json"
        deathmatch = DeathmatchPool(
            artists,
            main_pool,
            deathmatch_file,
        )

        self.assertCountEqual(deathmatch.enqueue(["a", "b"]), ["a", "b"])
        self.assertNotIn("a", main_pool.pool)
        self.assertNotIn("b", main_pool.pool)
        self.assertEqual(main_pool.get_pool_stats()["out_count"], 0)
        self.assertEqual(elo.get_rating("a"), 1725)
        self.assertEqual(elo.get_rating("b"), 1280)

        reloaded = DeathmatchPool(
            artists,
            main_pool,
            deathmatch_file,
        )
        pair_a, pair_b = reloaded.select_pair()
        self.assertEqual(len(pair_a), 1)
        self.assertEqual(len(pair_b), 1)
        self.assertFalse(set(pair_a) & set(pair_b))

        self.assertTrue(reloaded.resolve("a", keep=True))
        self.assertIn("a", main_pool.pool)
        self.assertNotIn("a", main_pool.manual_excluded)
        self.assertEqual(elo.get_rating("a"), 1725)

        self.assertTrue(reloaded.resolve("b", keep=False))
        self.assertNotIn("b", main_pool.pool)
        self.assertIn("b", main_pool.manual_excluded)
        self.assertEqual(main_pool.get_pool_stats()["out_count"], 1)
        self.assertEqual(reloaded.artists, [])

    def test_deathmatch_first_run_migrates_old_heart_exclusions_once(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        artists = ["old_heart", "active"]
        elo = ELOSystem(ratings={"old_heart": 1325})
        main_pool = ActivePool(
            artists,
            elo,
            pool_size=2,
            pool_file=Path(temp_dir.name) / "active_pool.json",
        )
        main_pool.remove_artists(["old_heart"], permanent=True)
        deathmatch_file = Path(temp_dir.name) / "deathmatch.json"

        migrated = DeathmatchPool(artists, main_pool, deathmatch_file)
        self.assertEqual(migrated.artists, ["old_heart"])
        self.assertTrue(deathmatch_file.exists())

        self.assertTrue(migrated.resolve("old_heart", keep=False))
        restarted = DeathmatchPool(artists, main_pool, deathmatch_file)
        self.assertEqual(restarted.artists, [])
        self.assertIn("old_heart", main_pool.manual_excluded)

    def test_deathmatch_generation_allows_one_final_artist(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        tags_file = Path(temp_dir.name) / "artists.txt"
        tags_file.write_text("alpha\n", encoding="utf-8")
        manager = ArtistTagManager(
            tags_file,
            ELOSystem(),
            Path(temp_dir.name) / "temporary_pool.json",
        )

        with patch(
            "artist_elo_ranker.generate_image",
            new=AsyncMock(return_value=True),
        ) as mocked_generate:
            path_a, path_b = asyncio.run(generate_deathmatch_pair(
                "portrait, {artist_placeholder}",
                manager,
                Mock(),
                Path(temp_dir.name),
                GenerationSettings(),
                42,
                ["alpha"],
                [],
            ))

        self.assertIsNotNone(path_a)
        self.assertIsNone(path_b)
        self.assertEqual(mocked_generate.await_count, 1)

    def test_generation_settings_round_trip_all_values(self):
        settings = GenerationSettings.from_values(
            "normal_portrait",
            31,
            6.4,
            "k_dpmpp_2m",
            "123456",
            True,
            0.24,
            "exponential",
            False,
            2,
        )

        restored = GenerationSettings.from_dict(settings.to_dict())

        self.assertEqual(restored, settings)
        self.assertEqual(restored.dimension_text, "832 × 1216")

    def test_prompt_preset_persists_prompt_negative_and_all_settings(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        filepath = Path(temp_dir.name) / "prompt_presets.json"
        settings = GenerationSettings.from_values(
            "normal_landscape",
            28,
            5.5,
            "k_euler_ancestral",
            "42",
            True,
            0.1,
            "karras",
            True,
            3,
        )

        store = PromptPresetStore(filepath)
        store.save_slot(10, "portrait, {artist_placeholder}", "text", settings)
        reloaded = PromptPresetStore(filepath)
        saved = reloaded.load_slot("10")

        self.assertEqual(saved["prompt"], "portrait, {artist_placeholder}")
        self.assertEqual(saved["negative_prompt"], "text")
        self.assertEqual(
            GenerationSettings.from_dict(saved["settings"]),
            settings,
        )

    def test_comparison_pair_passes_one_shared_seed_to_both_images(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        manager = Mock()
        manager.get_comparison_pair.return_value = (
            ["artist_a"],
            ["artist_b"],
            POOL_ACTION_TRIM_TO_150,
        )
        manager.format_artist_tags.side_effect = lambda artists: ", ".join(artists)
        settings = GenerationSettings()

        with patch(
            "artist_elo_ranker.generate_image",
            new=AsyncMock(return_value=True),
        ) as mocked_generate:
            result = asyncio.run(
                generate_comparison_pair(
                    "1girl, {artist_placeholder}",
                    manager,
                    Mock(),
                    Path(temp_dir.name),
                    settings,
                    987654,
                    "bad anatomy",
                )
            )

        self.assertEqual(result[-1], POOL_ACTION_TRIM_TO_150)
        self.assertEqual(mocked_generate.await_count, 2)
        self.assertEqual(mocked_generate.await_args_list[0].args[4], 987654)
        self.assertEqual(mocked_generate.await_args_list[1].args[4], 987654)

    def test_generate_image_applies_every_supported_setting(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        output_path = Path(temp_dir.name) / "generated.png"
        settings = GenerationSettings.from_values(
            "normal_portrait",
            33,
            6.6,
            "k_dpmpp_2m",
            None,
            True,
            0.22,
            "polyexponential",
            False,
            -1,
        )
        captured = []

        async def fake_request(instance, session):
            captured.append(instance)
            return Mock(files=[("generated.png", b"image-bytes")])

        with patch(
            "artist_elo_ranker.GenerateImageInfer.request",
            new=fake_request,
        ):
            success = asyncio.run(
                generate_image(
                    Mock(),
                    "1girl, artist: example",
                    output_path,
                    settings,
                    7654321,
                    "bad anatomy",
                )
            )

        self.assertTrue(success)
        self.assertEqual(output_path.read_bytes(), b"image-bytes")
        params = captured[0].parameters
        self.assertEqual((params.width, params.height), (832, 1216))
        self.assertEqual(params.steps, 33)
        self.assertEqual(params.seed, 7654321)
        self.assertEqual(params.scale, 6.6)
        self.assertEqual(params.cfg_rescale, 0.22)
        self.assertIsNone(params.ucPreset)
        self.assertFalse(params.qualityToggle)
        self.assertIsNotNone(params.skip_cfg_above_sigma)
        self.assertEqual(params.noise_schedule.value, "polyexponential")

    def test_anlas_balance_is_read_from_subscription_response(self):
        ranker = ArtistELORanker.__new__(ArtistELORanker)
        ranker.anlas_balance = None
        ranker.get_session = Mock(return_value=Mock())

        with patch(
            "artist_elo_ranker.Subscription.request",
            new=AsyncMock(return_value=Mock(anlas_left=10006)),
        ):
            balance = asyncio.run(ranker.refresh_anlas_balance())

        self.assertEqual(balance, 10006)
        self.assertIn("10,006", ranker.format_anlas_display())


if __name__ == "__main__":
    unittest.main()
