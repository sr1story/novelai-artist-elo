#!/usr/bin/env python3
"""
Artist ELO Ranking System for NovelAI Image Generation

A blind comparison system that generates images with random artist tag combinations
(1-3 artists) and allows users to pick their preferred image. Artists gain/lose ELO
based on the outcomes.
"""

import asyncio
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from pydantic import SecretStr

from novelai_python import GenerateImageInfer, ImageGenerateResp, ApiCredential
from novelai_python.sdk.ai.generate_image import Model, Sampler, UCPreset

# Import configuration
from config import (
    get_api_key,
    ARTIST_TAGS_FILE,
    DATA_DIR,
    ELO_RATINGS_FILE,
    COMPARISON_IMAGES_DIR,
    COMPARISON_HISTORY_FILE,
    ACTIVE_POOL_FILE,
    CURRENT_COMPARISON_FILE,
    STEPS,
    IMG_WIDTH,
    IMG_HEIGHT,
    DEFAULT_ELO,
    K_FACTOR,
    ACTIVE_POOL_SIZE,
    NEW_ARTIST_PROBABILITY,
    LOSER_ROTATION_PROBABILITY,
    SERVER_HOST,
    SERVER_PORT,
    APP_USERNAME,
    APP_PASSWORD,
    INBROWSER,
    NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
)

# --------------------------------------------------------------------------------
# NovelAI Model Configuration
# --------------------------------------------------------------------------------

MODEL = Model.NAI_DIFFUSION_4_5_FULL
SAMPLER = Sampler.K_EULER_ANCESTRAL
UC_PRESET = UCPreset.TYPE0


# --------------------------------------------------------------------------------
# Mobile-first UI
# --------------------------------------------------------------------------------

APP_HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#111827">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Artist ELO">
"""

MOBILE_CSS = """
.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding-bottom: calc(7rem + env(safe-area-inset-bottom)) !important;
}

#app-header h1 {
    margin-bottom: .25rem;
    letter-spacing: -.03em;
}

#app-header p {
    margin-top: 0;
    color: var(--body-text-color-subdued);
}

#status-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    padding: .7rem 1rem;
    background: var(--background-fill-secondary);
}

#comparison-row {
    gap: 1rem;
}

.image-card {
    min-width: 0 !important;
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    padding: .65rem;
    background: var(--block-background-fill);
    box-shadow: var(--block-shadow);
}

.comparison-image img {
    width: 100% !important;
    height: auto !important;
    aspect-ratio: 1 / 1;
    object-fit: contain !important;
    border-radius: 12px;
}

#vote-dock {
    position: sticky;
    bottom: max(.5rem, env(safe-area-inset-bottom));
    z-index: 20;
    gap: .6rem;
    margin-top: .75rem;
    padding: .7rem;
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    background: color-mix(in srgb, var(--background-fill-primary) 92%, transparent);
    backdrop-filter: blur(14px);
    box-shadow: 0 12px 32px rgba(0, 0, 0, .18);
}

#vote-a, #vote-b {
    min-height: 54px;
    font-size: 1rem;
    font-weight: 750;
}

#secondary-actions button {
    min-height: 46px;
}

@media (max-width: 760px) {
    .gradio-container {
        padding: .7rem .7rem calc(6.5rem + env(safe-area-inset-bottom)) !important;
    }

    #app-header h1 {
        font-size: 1.55rem;
    }

    #desktop-help {
        display: none;
    }

    #main-layout,
    #comparison-row {
        flex-direction: column !important;
    }

    #main-layout > div,
    #comparison-row > div {
        width: 100% !important;
        min-width: 0 !important;
    }

    .image-card {
        padding: .45rem;
        border-radius: 14px;
    }

    #vote-dock {
        margin-left: -.25rem;
        margin-right: -.25rem;
        padding: .55rem;
        border-radius: 16px;
    }

    #vote-dock button {
        min-width: 0 !important;
    }
}
"""


# --------------------------------------------------------------------------------
# Ranking direction presets
# --------------------------------------------------------------------------------

RANKING_MODE_BALANCED = "balanced"
RANKING_MODE_NEWCOMERS = "newcomers"
RANKING_MODE_TOP = "top"
RANKING_MODE_FAST_ROTATION = "fast_rotation"

RANKING_MODE_CONFIG = {
    RANKING_MODE_BALANCED: {
        "label": "균형 · 기존",
        "description": "기존 방식입니다. 비교 횟수가 적은 작가를 완만하게 우선합니다.",
        "removal_probability": 0.15,
        "addition_probability": 0.15,
    },
    RANKING_MODE_NEWCOMERS: {
        "label": "신규 탐색",
        "description": "비교 5회 미만 작가와 아직 평가하지 않은 작가를 더 자주 보여줍니다.",
        "removal_probability": 0.28,
        "addition_probability": 0.28,
    },
    RANKING_MODE_TOP: {
        "label": "상위권 정밀화",
        "description": "비교 5회 이상인 상위 30% 작가를 더 자주 비교합니다.",
        "removal_probability": 0.08,
        "addition_probability": 0.08,
    },
    RANKING_MODE_FAST_ROTATION: {
        "label": "빠른 교체",
        "description": "충분히 평가됐지만 평균보다 낮은 작가를 재검증하고 풀 교체를 빠르게 합니다.",
        "removal_probability": 0.35,
        "addition_probability": 0.35,
    },
}

RANKING_MODE_CHOICES = [
    (settings["label"], mode)
    for mode, settings in RANKING_MODE_CONFIG.items()
]
MODE_FOCUS_PROBABILITY = 0.70
MIN_CONFIDENT_COMPARISONS = 5
TOP_ARTIST_FRACTION = 0.30


def normalize_ranking_mode(mode: str) -> str:
    """Return a supported ranking mode, falling back to the existing strategy."""
    return mode if mode in RANKING_MODE_CONFIG else RANKING_MODE_BALANCED


def get_ranking_mode_label(mode: str) -> str:
    """Return the user-facing label for a ranking mode."""
    return RANKING_MODE_CONFIG[normalize_ranking_mode(mode)]["label"]


def get_ranking_mode_description(mode: str) -> str:
    """Return the short explanation shown below the mode dropdown."""
    return RANKING_MODE_CONFIG[normalize_ranking_mode(mode)]["description"]


# --------------------------------------------------------------------------------
# ELO Rating System
# --------------------------------------------------------------------------------

@dataclass
class ELOSystem:
    """Manages ELO ratings for artists."""
    ratings: dict = field(default_factory=dict)
    comparison_count: int = 0
    artist_comparisons: dict = field(default_factory=dict)  # Track per-artist comparisons

    @classmethod
    def load(cls, filepath: Path) -> "ELOSystem":
        """Load ELO ratings from file."""
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                system = cls()
                system.ratings = data.get("ratings", {})
                system.comparison_count = data.get("comparison_count", 0)
                system.artist_comparisons = data.get("artist_comparisons", {})
                return system
        return cls()

    def save(self, filepath: Path):
        """Save ELO ratings to file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "ratings": self.ratings,
                "comparison_count": self.comparison_count,
                "artist_comparisons": self.artist_comparisons
            }, f, indent=2)

    def get_rating(self, artist: str) -> float:
        """Get ELO rating for an artist, defaulting to DEFAULT_ELO."""
        return self.ratings.get(artist, DEFAULT_ELO)

    def get_combined_rating(self, artists: List[str]) -> float:
        """Get average ELO rating for a combination of artists."""
        if not artists:
            return DEFAULT_ELO
        return sum(self.get_rating(a) for a in artists) / len(artists)

    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, winners: List[str], losers: List[str]):
        """
        Update ELO ratings after a comparison.
        Uses INDIVIDUAL-based calculation: each artist's gain/loss is based on
        their own ELO vs the opposing team's average (not team vs team).

        Scaled to maintain zero-sum: total ELO gained = total ELO lost.
        """
        # Find overlapping artists (they're neutral - no ELO change)
        overlap = set(winners) & set(losers)
        actual_winners = [a for a in winners if a not in overlap]
        actual_losers = [a for a in losers if a not in overlap]

        if overlap:
            print(f"Overlap detected (neutral): {overlap}")

        if not actual_winners or not actual_losers:
            self.comparison_count += 1
            return

        # Get opposing team averages for individual calculations
        winner_team_avg = self.get_combined_rating(winners)
        loser_team_avg = self.get_combined_rating(losers)

        # Calculate raw changes for winners
        winner_changes = []
        for artist in actual_winners:
            current = self.get_rating(artist)
            expected = self.calculate_expected_score(current, loser_team_avg)
            change = K_FACTOR * (1 - expected)
            winner_changes.append((artist, change))

        # Calculate raw changes for losers
        loser_changes = []
        for artist in actual_losers:
            current = self.get_rating(artist)
            expected = self.calculate_expected_score(current, winner_team_avg)
            change = K_FACTOR * (0 - (1 - expected))
            loser_changes.append((artist, change))

        # Scale to maintain zero-sum
        total_winner_gain = sum(c for _, c in winner_changes)
        total_loser_loss = sum(c for _, c in loser_changes)  # negative

        # Scale loser losses so total loss = total gain
        if total_loser_loss != 0:
            scale_factor = -total_winner_gain / total_loser_loss
        else:
            scale_factor = 1.0

        # Apply changes
        for artist, change in winner_changes:
            self.ratings[artist] = self.get_rating(artist) + change
            self.artist_comparisons[artist] = self.artist_comparisons.get(artist, 0) + 1

        for artist, change in loser_changes:
            scaled_change = change * scale_factor
            self.ratings[artist] = self.get_rating(artist) + scaled_change
            self.artist_comparisons[artist] = self.artist_comparisons.get(artist, 0) + 1

        self.comparison_count += 1

    def get_top_artists(self, n: int = 50) -> List[Tuple[str, float, int]]:
        """Get top N artists by ELO rating with their comparison counts."""
        sorted_artists = sorted(
            self.ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        return [(artist, rating, self.get_artist_comparison_count(artist))
                for artist, rating in sorted_artists]

    def get_bottom_artists(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get bottom N artists by ELO rating."""
        sorted_artists = sorted(
            self.ratings.items(),
            key=lambda x: x[1]
        )[:n]
        return sorted_artists

    def get_artist_comparison_count(self, artist: str) -> int:
        """Get the number of comparisons an artist has participated in."""
        return self.artist_comparisons.get(artist, 0)


# --------------------------------------------------------------------------------
# Active Pool System
# --------------------------------------------------------------------------------

class ActivePool:
    """
    Manages a smaller active pool of artists for more meaningful comparisons.

    Strategy:
    - Maintain a pool of ~150 artists that get compared frequently
    - Winners stay in the pool (good artists get more comparisons)
    - Losers may get rotated out (with some probability)
    - Periodically introduce new random artists to discover hidden gems
    - Weight selection according to the selected ranking direction preset
    - Keep 30% of selections on the balanced strategy to preserve coverage
    """

    def __init__(self, all_artists: List[str], elo_system: ELOSystem,
                 pool_size: int = ACTIVE_POOL_SIZE, pool_file: Path = None):
        self.all_artists = all_artists
        self.elo_system = elo_system
        self.pool_size = pool_size
        self.pool_file = pool_file or ACTIVE_POOL_FILE
        self.pool: List[str] = []
        self.ranking_mode = RANKING_MODE_BALANCED
        self.load()

    def load(self):
        """Load pool from file or initialize if not exists."""
        if self.pool_file.exists():
            with open(self.pool_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.pool = data.get("pool", [])
                self.ranking_mode = normalize_ranking_mode(
                    data.get("ranking_mode", RANKING_MODE_BALANCED)
                )
                # Validate pool members still exist in all_artists
                self.pool = [a for a in self.pool if a in self.all_artists]

        # Initialize or refill pool if needed
        if len(self.pool) < self.pool_size:
            self._refill_pool()

    def save(self):
        """Save pool to file."""
        with open(self.pool_file, "w", encoding="utf-8") as f:
            json.dump({
                "pool": self.pool,
                "ranking_mode": self.ranking_mode,
            }, f, indent=2)

    def set_ranking_mode(self, mode: str) -> str:
        """Set and persist the ranking direction preset."""
        normalized = normalize_ranking_mode(mode)
        if normalized != self.ranking_mode:
            self.ranking_mode = normalized
            self.save()
        return self.ranking_mode

    def get_ranking_mode(self) -> str:
        """Return the active ranking direction preset."""
        return self.ranking_mode

    def _refill_pool(self):
        """Fill pool up to pool_size with random artists not already in pool."""
        available = [a for a in self.all_artists if a not in self.pool]
        needed = self.pool_size - len(self.pool)
        if needed > 0 and available:
            new_artists = random.sample(available, min(needed, len(available)))
            self.pool.extend(new_artists)
            print(f"Added {len(new_artists)} new artists to active pool. Pool size: {len(self.pool)}")
        self.save()

    def get_selection_weight(self, artist: str) -> float:
        """
        Calculate selection weight for an artist.
        Artists with fewer comparisons get higher weight (need more data).
        """
        comparisons = self.elo_system.get_artist_comparison_count(artist)
        # Inverse weight: fewer comparisons = higher weight
        # Add 1 to avoid division by zero, use sqrt to dampen the effect
        return 1.0 / (1.0 + (comparisons ** 0.5))

    @staticmethod
    def _weighted_choice(candidates: List[str], weights: List[float]) -> str:
        """Select one candidate using non-negative weights."""
        total_weight = sum(weights)
        if total_weight <= 0:
            return random.choice(candidates)

        target = random.uniform(0, total_weight)
        cumulative = 0.0
        for artist, weight in zip(candidates, weights):
            effective_weight = max(0.0, weight)
            if effective_weight == 0:
                continue
            cumulative += effective_weight
            if target <= cumulative:
                return artist
        return candidates[-1]

    def _get_focus_candidates(self, candidates: List[str]) -> List[str]:
        """Return the artists emphasized by the active direction preset."""
        if self.ranking_mode == RANKING_MODE_NEWCOMERS:
            return [
                artist for artist in candidates
                if self.elo_system.get_artist_comparison_count(artist)
                < MIN_CONFIDENT_COMPARISONS
            ]

        confident = [
            artist for artist in candidates
            if self.elo_system.get_artist_comparison_count(artist)
            >= MIN_CONFIDENT_COMPARISONS
        ]
        if not confident:
            return []

        if self.ranking_mode == RANKING_MODE_TOP:
            top_count = max(1, math.ceil(len(confident) * TOP_ARTIST_FRACTION))
            return sorted(
                confident,
                key=lambda artist: self.elo_system.get_rating(artist),
                reverse=True,
            )[:top_count]

        if self.ranking_mode == RANKING_MODE_FAST_ROTATION:
            pool_average = sum(
                self.elo_system.get_rating(artist) for artist in self.pool
            ) / len(self.pool)
            return [
                artist for artist in confident
                if self.elo_system.get_rating(artist) < pool_average
            ]

        return []

    def _select_from_candidates(self, candidates: List[str]) -> str:
        """Select an artist with a 70/30 preset-to-balanced mixture."""
        selection_pool = candidates
        if self.ranking_mode != RANKING_MODE_BALANCED:
            focus_candidates = self._get_focus_candidates(candidates)
            if focus_candidates and random.random() < MODE_FOCUS_PROBABILITY:
                selection_pool = focus_candidates

        weights = [self.get_selection_weight(artist) for artist in selection_pool]
        return self._weighted_choice(selection_pool, weights)

    def select_artist(self) -> str:
        """Select a single artist according to the active direction preset."""
        if not self.pool:
            self._refill_pool()
        return self._select_from_candidates(self.pool)

    def select_combination(self, min_artists: int = 1, max_artists: int = 3) -> List[str]:
        """Select a combination of 1-3 artists from the pool."""
        if not self.pool:
            self._refill_pool()

        num_artists = random.randint(min_artists, max_artists)
        num_artists = min(num_artists, len(self.pool))

        selected = []
        pool_copy = self.pool.copy()

        for _ in range(num_artists):
            if not pool_copy:
                break
            artist = self._select_from_candidates(pool_copy)
            selected.append(artist)
            pool_copy.remove(artist)

        # Shuffle to randomize tag order in prompt
        random.shuffle(selected)
        return selected

    def process_result(self, winners: List[str], losers: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float, bool]]]:
        """
        Process comparison result to update the pool.
        Uses weighted removal from entire pool (not just losers).

        Returns: (rotated_out, rotated_in) where each is list of (artist, elo, [is_returning])
        """
        rotated_out = []  # [(artist, elo), ...]
        rotated_in = []   # [(artist, elo, is_returning), ...]

        # Determine probabilities based on pool size. Around the configured
        # target, the active direction preset controls turnover speed.
        pool_diff = len(self.pool) - self.pool_size
        if pool_diff > 10:  # Pool too big
            removal_prob = 0.3
            addition_prob = 0.05
        elif pool_diff < -10:  # Pool too small
            removal_prob = 0.05
            addition_prob = 0.3
        else:  # Around target
            mode_config = RANKING_MODE_CONFIG[self.ranking_mode]
            removal_prob = mode_config["removal_probability"]
            addition_prob = mode_config["addition_probability"]

        # Weighted removal from entire pool (not just losers)
        # Weight = confidence(matches) * underperformance(relative to pool max)
        if random.random() < removal_prob and len(self.pool) > self.pool_size // 2:
            pool_elos = [self.elo_system.get_rating(a) for a in self.pool]
            pool_max_elo = max(pool_elos)

            removal_weights = []
            for artist in self.pool:
                matches = self.elo_system.get_artist_comparison_count(artist)
                elo = self.elo_system.get_rating(artist)

                # Newcomers are protected until they have enough comparisons.
                confidence = (
                    1.0 if matches >= MIN_CONFIDENT_COMPARISONS else 0.0
                )

                # Underperformance: relative to pool's best performer (squared)
                # Squared so worst performers are MUCH more likely to be removed
                # Mirrors the squared addition weighting for symmetry
                underperformance = ((pool_max_elo - elo) / 100.0) ** 2

                weight = confidence * underperformance
                removal_weights.append(weight)

            total_weight = sum(removal_weights)
            if total_weight > 0:
                r = random.uniform(0, total_weight)
                cumulative = 0
                for artist, weight in zip(self.pool, removal_weights):
                    if weight <= 0:
                        continue
                    cumulative += weight
                    if r <= cumulative:
                        elo = self.elo_system.get_rating(artist)
                        self.pool.remove(artist)
                        rotated_out.append((artist, elo))
                        print(f"Rotated out: {artist} (ELO: {elo:.0f})")
                        break

        # Maybe introduce new artist, weighted by ELO (squared for stronger preference)
        # Higher ELO = much higher chance of being added back
        if random.random() < addition_prob:
            available = [a for a in self.all_artists if a not in self.pool]
            if available:
                # Newcomer mode spends 70% of its focused additions on artists
                # that have never received an ELO result. The remaining 30%
                # uses the existing high-ELO return weighting.
                if (
                    self.ranking_mode == RANKING_MODE_NEWCOMERS
                    and random.random() < MODE_FOCUS_PROBABILITY
                ):
                    fresh = [a for a in available if a not in self.elo_system.ratings]
                    if fresh:
                        available = fresh

                elos = [self.elo_system.get_rating(a) for a in available]
                min_elo = min(elos)
                # Square the weight difference for stronger high-ELO preference
                # 1700 vs 1300: old=(500 vs 100)=5x, new=(500^2 vs 100^2)=25x
                weights = [((e - min_elo + 100) ** 2) for e in elos]
                total = sum(weights)
                r = random.uniform(0, total)
                cumulative = 0
                new_artist = available[-1]
                for a, w in zip(available, weights):
                    cumulative += w
                    if r <= cumulative:
                        new_artist = a
                        break
                self.pool.append(new_artist)
                elo = self.elo_system.get_rating(new_artist)
                is_returning = new_artist in self.elo_system.ratings
                rotated_in.append((new_artist, elo, is_returning))
                status = f"returning, ELO: {elo:.0f}" if is_returning else "fresh"
                print(f"Rotated in: {new_artist} ({status})")

        # Hard cap: if pool exceeds target + 20, force remove lowest performers
        # This ensures pool size stays bounded
        max_pool_size = self.pool_size + 20
        while len(self.pool) > max_pool_size:
            # Find artists with enough matches to judge (confidence >= 1)
            candidates = [(a, self.elo_system.get_rating(a))
                          for a in self.pool
                          if self.elo_system.get_artist_comparison_count(a) >= 5]
            if candidates:
                # Remove lowest ELO among confident artists
                worst = min(candidates, key=lambda x: x[1])
                self.pool.remove(worst[0])
                rotated_out.append((worst[0], worst[1]))
                print(f"Hard cap removal: {worst[0]} (ELO: {worst[1]:.0f})")
            else:
                # No confident artists, remove lowest ELO anyway
                worst = min(self.pool, key=lambda a: self.elo_system.get_rating(a))
                elo = self.elo_system.get_rating(worst)
                self.pool.remove(worst)
                rotated_out.append((worst, elo))
                print(f"Hard cap removal (low confidence): {worst} (ELO: {elo:.0f})")

        # Ensure pool doesn't get too small
        if len(self.pool) < self.pool_size // 2:
            self._refill_pool()

        self.save()
        return rotated_out, rotated_in

    def restore_artists(self, artists: List[str]):
        """Restore artists to the pool (for undo)."""
        for artist in artists:
            if artist not in self.pool and artist in self.all_artists:
                self.pool.append(artist)
        self.save()

    def revert_rotation(self, rotated_out: List[str], rotated_in: List[str]):
        """Undo every pool membership change made by the latest comparison."""
        for artist in rotated_in:
            if artist in self.pool:
                self.pool.remove(artist)
        for artist in rotated_out:
            if artist not in self.pool and artist in self.all_artists:
                self.pool.append(artist)
        self.save()

    def get_pool_stats(self) -> dict:
        """Get statistics about the current pool."""
        if not self.pool:
            return {"size": 0, "avg_comparisons": 0, "avg_elo": DEFAULT_ELO,
                    "at_risk": [], "newcomers": 0, "safe": 0, "total_artists": 0}

        comparisons = [self.elo_system.get_artist_comparison_count(a) for a in self.pool]
        elos = [self.elo_system.get_rating(a) for a in self.pool]
        pool_max_elo = max(elos) if elos else DEFAULT_ELO
        pool_avg_elo = sum(elos) / len(elos) if elos else DEFAULT_ELO

        # Categorize artists relative to pool
        at_risk = []  # Lower ELO in pool + enough matches
        newcomers = 0  # < 5 matches (protected)
        safe = 0  # Top performers (above pool average)

        for artist in self.pool:
            matches = self.elo_system.get_artist_comparison_count(artist)
            elo = self.elo_system.get_rating(artist)

            if matches < 5:
                newcomers += 1
            elif elo >= pool_avg_elo:
                safe += 1
            else:
                # At risk: has enough matches AND below pool average
                # Calculate removal weight for sorting (relative to pool max)
                confidence = min(1.0, matches / 5.0)
                underperformance = (pool_max_elo - elo) / 100.0
                weight = confidence * underperformance
                at_risk.append((artist, elo, matches, weight))

        # Sort at_risk by weight (most likely to be removed first)
        at_risk.sort(key=lambda x: x[3], reverse=True)

        return {
            "size": len(self.pool),
            "avg_comparisons": sum(comparisons) / len(comparisons),
            "min_comparisons": min(comparisons),
            "max_comparisons": max(comparisons),
            "avg_elo": sum(elos) / len(elos),
            "total_artists": len(self.all_artists),
            "at_risk": at_risk[:10],  # Top 10 most at risk
            "at_risk_count": len(at_risk),
            "newcomers": newcomers,
            "safe": safe,
            "ranking_mode": self.ranking_mode,
        }


# --------------------------------------------------------------------------------
# Artist Tag Management
# --------------------------------------------------------------------------------

class ArtistTagManager:
    """Manages loading and selecting artist tags."""

    def __init__(self, tags_file: Path, elo_system: ELOSystem = None):
        self.tags_file = tags_file
        self.artists: List[str] = []
        self.elo_system = elo_system
        self.active_pool: Optional[ActivePool] = None
        self.load_artists()

    def load_artists(self):
        """Load artist tags from file."""
        if self.tags_file.exists():
            with open(self.tags_file, "r", encoding="utf-8") as f:
                self.artists = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(self.artists)} artist tags")
        else:
            print(f"Warning: Artist tags file not found: {self.tags_file}")
            self.artists = []

    def initialize_pool(self, elo_system: ELOSystem):
        """Initialize the active pool with the ELO system."""
        self.elo_system = elo_system
        self.active_pool = ActivePool(self.artists, elo_system)

    def get_random_combination(self, min_artists: int = 1, max_artists: int = 3) -> List[str]:
        """Get a random combination of 1-3 artists, using active pool if available."""
        if self.active_pool:
            return self.active_pool.select_combination(min_artists, max_artists)

        # Fallback to pure random if no pool
        if not self.artists:
            return []
        num_artists = random.randint(min_artists, max_artists)
        return random.sample(self.artists, min(num_artists, len(self.artists)))

    def process_result(self, winners: List[str], losers: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float, bool]]]:
        """Process comparison result to update the active pool. Returns (rotated_out, rotated_in)."""
        if self.active_pool:
            return self.active_pool.process_result(winners, losers)
        return [], []

    def set_ranking_mode(self, mode: str) -> str:
        """Set and persist the active ranking direction preset."""
        if self.active_pool:
            return self.active_pool.set_ranking_mode(mode)
        return normalize_ranking_mode(mode)

    def get_ranking_mode(self) -> str:
        """Return the active ranking direction preset."""
        if self.active_pool:
            return self.active_pool.get_ranking_mode()
        return RANKING_MODE_BALANCED

    def restore_artists(self, artists: List[str]):
        """Restore artists to the pool (for undo)."""
        if self.active_pool:
            self.active_pool.restore_artists(artists)

    def revert_rotation(self, rotated_out: List[str], rotated_in: List[str]):
        """Undo pool additions and removals from the latest comparison."""
        if self.active_pool:
            self.active_pool.revert_rotation(rotated_out, rotated_in)

    def get_pool_stats(self) -> dict:
        """Get active pool statistics."""
        if self.active_pool:
            return self.active_pool.get_pool_stats()
        return {"size": 0, "total_artists": len(self.artists)}

    def format_artist_tags(self, artists: List[str]) -> str:
        """Format artist list as comma-separated artist tags."""
        return ", ".join(f"artist: {artist}" for artist in artists)


# --------------------------------------------------------------------------------
# Prompt Processing
# --------------------------------------------------------------------------------

def is_artist_tag(segment: str) -> bool:
    """
    Check if a segment is an artist tag.
    Handles: artist: name, artist:name, [artist: name], {artist: name:1.5}, etc.
    """
    stripped = segment.strip().lower()
    # Remove surrounding brackets
    stripped_no_brackets = stripped.strip("[]{}()")

    # Check for artist: or artist (space) patterns
    if stripped_no_brackets.startswith("artist:") or stripped_no_brackets.startswith("artist "):
        return True
    # Check for artist: anywhere (handles weighted like "artist:name:1.5")
    if "artist:" in stripped_no_brackets:
        return True
    # Check for "artist name" pattern at start
    if re.match(r"^\s*artist\s+\w", stripped_no_brackets):
        return True
    return False


def remove_artist_tags_with_position(prompt: str) -> tuple:
    """
    Remove existing artist tags from a prompt and return position of first one found.
    Returns: (cleaned_prompt, first_artist_index or -1 if none found)

    Handles:
    - artist: name, artist:name
    - [artist: name], {artist: name}, (artist: name)
    - artist:name:1.5 (weighted tags)
    """
    segments = prompt.split(",")
    filtered_segments = []
    first_artist_idx = -1

    for i, segment in enumerate(segments):
        if is_artist_tag(segment):
            if first_artist_idx == -1:
                first_artist_idx = len(filtered_segments)  # Position in filtered list
            continue
        filtered_segments.append(segment)

    return ",".join(filtered_segments), first_artist_idx


def remove_artist_tags(prompt: str) -> str:
    """Remove all artist tags from prompt, return cleaned prompt."""
    cleaned, _ = remove_artist_tags_with_position(prompt)
    return cleaned


def insert_artist_tags(prompt: str, artist_tags: str) -> str:
    """
    Insert artist tags into the prompt.
    If {artist_placeholder} exists, replace it.
    Otherwise, insert at position of first existing artist tag, or at end.
    """
    if "{artist_placeholder}" in prompt:
        return prompt.replace("{artist_placeholder}", artist_tags)

    # Remove any existing artist tags and get position of first one
    clean_prompt, first_artist_idx = remove_artist_tags_with_position(prompt)
    segments = clean_prompt.split(",")

    # If there was an existing artist tag, insert at that position
    if first_artist_idx >= 0 and first_artist_idx <= len(segments):
        segments.insert(first_artist_idx, f" {artist_tags}")
        return ",".join(segments)

    # No existing artist tags - always append at end
    return clean_prompt + ", " + artist_tags


# --------------------------------------------------------------------------------
# Image Generation
# --------------------------------------------------------------------------------

async def generate_image(
    session: ApiCredential,
    prompt: str,
    output_path: Path,
    negative_prompt: str = None,
    quality_toggle: bool = True,
    uc_preset: int = 0
) -> bool:
    """Generate a single image and save it."""
    try:
        # Map UC preset index to enum (-1 = None/disabled)
        uc_preset_map = {
            -1: None,  # Disabled
            0: UCPreset.TYPE0,
            1: UCPreset.TYPE1,
            2: UCPreset.TYPE2,
            3: UCPreset.TYPE3,
        }
        gen = GenerateImageInfer.build_generate(
            prompt=prompt,
            width=IMG_WIDTH,
            height=IMG_HEIGHT,
            model=MODEL,
            steps=STEPS,
            sampler=SAMPLER,
            negative_prompt=negative_prompt if negative_prompt else NEGATIVE_PROMPT,
            ucPreset=uc_preset_map.get(uc_preset, UCPreset.TYPE0),
            qualityToggle=quality_toggle,
            decrisp_mode=False,
            variety_boost=False,
        )

        resp = await gen.request(session=session)
        resp: ImageGenerateResp

        _, file_bytes = resp.files[0]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(file_bytes)

        return True
    except Exception as e:
        print(f"Error generating image: {e}")
        return False


async def generate_comparison_pair(
    base_prompt: str,
    artist_manager: ArtistTagManager,
    session: ApiCredential,
    output_dir: Path,
    negative_prompt: str = None,
    quality_toggle: bool = True,
    uc_preset: int = 0
) -> Tuple[Optional[Path], Optional[Path], List[str], List[str]]:
    """
    Generate two images with different artist combinations.
    Returns: (image_a_path, image_b_path, artists_a, artists_b)
    """
    # Get two different artist combinations (overlap is allowed, handled in ELO calc)
    artists_a = artist_manager.get_random_combination()
    artists_b = artist_manager.get_random_combination()

    # Ensure they're not identical (but overlap is fine)
    max_attempts = 50
    attempts = 0
    while set(artists_a) == set(artists_b) and attempts < max_attempts:
        artists_b = artist_manager.get_random_combination()
        attempts += 1

    # Format artist tags
    tags_a = artist_manager.format_artist_tags(artists_a)
    tags_b = artist_manager.format_artist_tags(artists_b)

    # Create prompts
    prompt_a = insert_artist_tags(base_prompt, tags_a)
    prompt_b = insert_artist_tags(base_prompt, tags_b)

    # Generate unique filenames
    timestamp = int(time.time() * 1000)
    path_a = output_dir / f"compare_{timestamp}_a.png"
    path_b = output_dir / f"compare_{timestamp}_b.png"

    print(f"Generating image A with artists: {artists_a}")
    print(f"Prompt A: {prompt_a[:200]}...")
    success_a = await generate_image(session, prompt_a, path_a, negative_prompt, quality_toggle, uc_preset)

    print(f"Generating image B with artists: {artists_b}")
    print(f"Prompt B: {prompt_b[:200]}...")
    success_b = await generate_image(session, prompt_b, path_b, negative_prompt, quality_toggle, uc_preset)

    if success_a and success_b:
        return path_a, path_b, artists_a, artists_b
    return None, None, [], []


# --------------------------------------------------------------------------------
# Comparison History
# --------------------------------------------------------------------------------

@dataclass
class ComparisonRecord:
    """Record of a single comparison."""
    timestamp: float
    artists_a: List[str]
    artists_b: List[str]
    winner: str  # "A" or "B"
    image_a_path: str
    image_b_path: str


class ComparisonHistory:
    """Manages comparison history."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.records: List[dict] = []
        self.load()

    def load(self):
        """Load history from file."""
        if self.filepath.exists():
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.records = json.load(f)

    def save(self):
        """Save history to file."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def add_record(self, record: ComparisonRecord):
        """Add a comparison record."""
        self.records.append({
            "timestamp": record.timestamp,
            "artists_a": record.artists_a,
            "artists_b": record.artists_b,
            "winner": record.winner,
            "image_a_path": record.image_a_path,
            "image_b_path": record.image_b_path
        })
        self.save()

    def get_artist_stats(self) -> dict:
        """
        Calculate stats for all artists from comparison history.
        Returns dict of artist -> {
            'rounds': total appearances,
            'wins': total wins,
            'solo': {'rounds': n, 'wins': n},
            'duo': {'rounds': n, 'wins': n},
            'trio': {'rounds': n, 'wins': n}
        }
        """
        stats = {}

        for record in self.records:
            artists_a = record.get("artists_a", [])
            artists_b = record.get("artists_b", [])
            winner = record.get("winner", "")

            # Determine group sizes
            size_a = len(artists_a)
            size_b = len(artists_b)

            # Process side A
            for artist in artists_a:
                if artist not in stats:
                    stats[artist] = {
                        'rounds': 0, 'wins': 0,
                        'solo': {'rounds': 0, 'wins': 0},
                        'duo': {'rounds': 0, 'wins': 0},
                        'trio': {'rounds': 0, 'wins': 0}
                    }
                stats[artist]['rounds'] += 1
                won = (winner == "A")
                if won:
                    stats[artist]['wins'] += 1

                # Track by group size
                size_key = {1: 'solo', 2: 'duo', 3: 'trio'}.get(size_a, 'trio')
                stats[artist][size_key]['rounds'] += 1
                if won:
                    stats[artist][size_key]['wins'] += 1

            # Process side B
            for artist in artists_b:
                if artist not in stats:
                    stats[artist] = {
                        'rounds': 0, 'wins': 0,
                        'solo': {'rounds': 0, 'wins': 0},
                        'duo': {'rounds': 0, 'wins': 0},
                        'trio': {'rounds': 0, 'wins': 0}
                    }
                stats[artist]['rounds'] += 1
                won = (winner == "B")
                if won:
                    stats[artist]['wins'] += 1

                # Track by group size
                size_key = {1: 'solo', 2: 'duo', 3: 'trio'}.get(size_b, 'trio')
                stats[artist][size_key]['rounds'] += 1
                if won:
                    stats[artist][size_key]['wins'] += 1

        return stats


# --------------------------------------------------------------------------------
# Gradio UI Application
# --------------------------------------------------------------------------------

@dataclass
class UndoState:
    """State needed to undo the last comparison."""
    winners: List[str]
    losers: List[str]
    old_ratings: dict  # artist -> old rating
    old_comparisons: dict  # artist -> old comparison count
    rotated_out: List[str]  # artists that were rotated out of pool
    rotated_in: List[str]  # artists that were added to the pool
    # Previous comparison images (for restoring on undo)
    prev_image_a: Optional[str] = None
    prev_image_b: Optional[str] = None
    prev_artists_a: List[str] = field(default_factory=list)
    prev_artists_b: List[str] = field(default_factory=list)


class ArtistELORanker:
    """Main application class."""

    def __init__(self):
        # DATA_DIR may point to a newly mounted cloud disk, so create it before
        # any of the JSON-backed managers attempt to save their initial state.
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        COMPARISON_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        self.elo_system = ELOSystem.load(ELO_RATINGS_FILE)
        self.artist_manager = ArtistTagManager(ARTIST_TAGS_FILE)
        # Initialize the active pool with ELO system
        self.artist_manager.initialize_pool(self.elo_system)
        self.history = ComparisonHistory(COMPARISON_HISTORY_FILE)
        self.session: Optional[ApiCredential] = None

        # Current comparison state
        self.current_image_a: Optional[Path] = None
        self.current_image_b: Optional[Path] = None
        self.current_artists_a: List[str] = []
        self.current_artists_b: List[str] = []
        self.current_prompt: str = DEFAULT_PROMPT
        self.current_custom_prompt: str = ""
        self.current_negative_prompt: str = ""
        self.current_quality_toggle: bool = True
        self.current_uc_preset: int = 0

        # Undo state
        self.last_undo_state: Optional[UndoState] = None
        self.selection_made: bool = False  # Track if a selection was made for current pair

        # Rotation log: list of (type, artist, elo, extra_info) - most recent first
        # type: "out" or "in", extra_info: is_returning for "in"
        self.rotation_log: List[Tuple[str, str, float, Optional[bool]]] = []

        # Reuse the last unexpired pair after a mobile page refresh or server
        # restart. This avoids an accidental extra NovelAI generation charge.
        self.load_current_comparison()

    @staticmethod
    def _load_image_path(value: str) -> Optional[Path]:
        """Return a saved comparison image path only when it is still safe and valid."""
        if not value:
            return None

        try:
            path = Path(value).expanduser().resolve()
            image_dir = COMPARISON_IMAGES_DIR.resolve()
        except (OSError, RuntimeError):
            return None

        if image_dir not in path.parents or not path.is_file():
            return None
        return path

    def load_current_comparison(self):
        """Restore the latest comparison so refreshing the phone does not regenerate it."""
        if not CURRENT_COMPARISON_FILE.exists():
            return

        try:
            with open(CURRENT_COMPARISON_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            image_a = self._load_image_path(data.get("image_a", ""))
            image_b = self._load_image_path(data.get("image_b", ""))
            artists_a = data.get("artists_a", [])
            artists_b = data.get("artists_b", [])

            if not image_a or not image_b or not artists_a or not artists_b:
                return

            self.current_image_a = image_a
            self.current_image_b = image_b
            self.current_artists_a = [str(a) for a in artists_a]
            self.current_artists_b = [str(a) for a in artists_b]
            self.current_prompt = str(data.get("processed_prompt", DEFAULT_PROMPT))
            self.current_custom_prompt = str(data.get("custom_prompt", ""))
            self.current_negative_prompt = str(data.get("negative_prompt", ""))
            self.current_quality_toggle = bool(data.get("quality_toggle", True))
            self.current_uc_preset = int(data.get("uc_preset", 0))
            self.selection_made = bool(data.get("selection_made", False))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"Could not restore current comparison: {exc}")

    def save_current_comparison(self):
        """Persist the current pair and form settings atomically."""
        if not self.current_image_a or not self.current_image_b:
            return

        data = {
            "image_a": str(self.current_image_a),
            "image_b": str(self.current_image_b),
            "artists_a": self.current_artists_a,
            "artists_b": self.current_artists_b,
            "processed_prompt": self.current_prompt,
            "custom_prompt": self.current_custom_prompt,
            "negative_prompt": self.current_negative_prompt,
            "quality_toggle": self.current_quality_toggle,
            "uc_preset": self.current_uc_preset,
            "selection_made": self.selection_made,
        }
        temp_file = CURRENT_COMPARISON_FILE.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_file.replace(CURRENT_COMPARISON_FILE)
        except OSError as exc:
            print(f"Could not save current comparison: {exc}")

    def get_session(self) -> ApiCredential:
        """Get or create API session."""
        if self.session is None:
            api_key = get_api_key()
            self.session = ApiCredential(api_token=SecretStr(api_key))
        return self.session

    def export_leaderboard_csv(self) -> str:
        """Export full leaderboard sorted by ELO as CSV with detailed stats."""
        sorted_artists = sorted(
            self.elo_system.ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        artist_stats = self.history.get_artist_stats()

        lines = ["Rank,Artist,ELO,Comparisons,Wins,Losses,WinRate,Solo_Rounds,Solo_Wins,Solo_WR,Duo_Rounds,Duo_Wins,Duo_WR,Trio_Rounds,Trio_Wins,Trio_WR"]

        for rank, (artist, rating) in enumerate(sorted_artists, 1):
            comparisons = self.elo_system.get_artist_comparison_count(artist)
            stats = artist_stats.get(artist, {})

            rounds = stats.get('rounds', 0)
            wins = stats.get('wins', 0)
            losses = rounds - wins
            win_rate = (wins / rounds * 100) if rounds > 0 else 0

            solo = stats.get('solo', {'rounds': 0, 'wins': 0})
            duo = stats.get('duo', {'rounds': 0, 'wins': 0})
            trio = stats.get('trio', {'rounds': 0, 'wins': 0})

            solo_wr = (solo['wins'] / solo['rounds'] * 100) if solo['rounds'] > 0 else 0
            duo_wr = (duo['wins'] / duo['rounds'] * 100) if duo['rounds'] > 0 else 0
            trio_wr = (trio['wins'] / trio['rounds'] * 100) if trio['rounds'] > 0 else 0

            lines.append(f"{rank},{artist},{rating:.0f},{comparisons},{wins},{losses},{win_rate:.1f},{solo['rounds']},{solo['wins']},{solo_wr:.1f},{duo['rounds']},{duo['wins']},{duo_wr:.1f},{trio['rounds']},{trio['wins']},{trio_wr:.1f}")

        return "\n".join(lines)

    def format_recent_history(self, limit: int = 10) -> str:
        """Format recent comparison history for display."""
        if not self.history.records:
            return "*아직 비교 기록이 없습니다.*"

        lines = ["*최신순:*", ""]
        recent = self.history.records[-limit:][::-1]  # Last N, reversed (newest first)

        for i, record in enumerate(recent, 1):
            winner = record.get("winner", "?")
            artists_a = record.get("artists_a", [])
            artists_b = record.get("artists_b", [])

            winner_artists = artists_a if winner == "A" else artists_b
            loser_artists = artists_b if winner == "A" else artists_a

            winner_str = ", ".join(winner_artists)
            loser_str = ", ".join(loser_artists)

            lines.append(f"{i}. **{winner_str}** 승 · {loser_str} 패")

        return "\n".join(lines)

    def format_top_artists_display(self) -> str:
        """Format top artists for display with win rate stats."""
        top_artists = self.elo_system.get_top_artists(30)
        pool_stats = self.artist_manager.get_pool_stats()
        artist_stats = self.history.get_artist_stats()

        lines = ["## 작가 랭킹", ""]

        if not top_artists:
            lines.append("아직 랭킹이 없습니다. 첫 비교를 시작하세요.")
        else:
            # Use markdown list format with win rate stats
            for i, (artist, rating, comparisons) in enumerate(top_artists, 1):
                stats = artist_stats.get(artist, {})
                rounds = stats.get('rounds', 0)
                wins = stats.get('wins', 0)
                wr = (wins / rounds * 100) if rounds > 0 else 0

                # Build compact W/R breakdown
                solo = stats.get('solo', {})
                duo = stats.get('duo', {})
                trio = stats.get('trio', {})

                # Format: S:80%(5) D:70%(10) - show W/R and count for each
                wr_parts = []
                if solo.get('rounds', 0) > 0:
                    solo_wr = solo['wins'] / solo['rounds'] * 100
                    wr_parts.append(f"S:{solo_wr:.0f}%({solo['rounds']})")
                if duo.get('rounds', 0) > 0:
                    duo_wr = duo['wins'] / duo['rounds'] * 100
                    wr_parts.append(f"D:{duo_wr:.0f}%({duo['rounds']})")
                if trio.get('rounds', 0) > 0:
                    trio_wr = trio['wins'] / trio['rounds'] * 100
                    wr_parts.append(f"T:{trio_wr:.0f}%({trio['rounds']})")

                wr_breakdown = f" {' '.join(wr_parts)}" if wr_parts else ""

                lines.append(f"{i}. **{artist}** {rating:.0f} — {wr:.0f}% ({rounds})")
                if wr_breakdown.strip():
                    lines.append(f"   {wr_breakdown.strip()}")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"**전체 비교:** {self.elo_system.comparison_count}  ")
        lines.append(f"**평가한 작가:** {len(self.elo_system.ratings)}  ")
        lines.append(f"**활성 풀:** {pool_stats.get('size', 0)}/{pool_stats.get('total_artists', 0)}  ")
        lines.append(
            f"**평가 방향:** {get_ranking_mode_label(self.artist_manager.get_ranking_mode())}"
        )

        # Pool health breakdown
        lines.append("")
        lines.append("---")
        lines.append("### 활성 풀 상태")
        safe = pool_stats.get('safe', 0)
        newcomers = pool_stats.get('newcomers', 0)
        at_risk_count = pool_stats.get('at_risk_count', 0)
        lines.append(f"평균 이상: {safe}  ")
        lines.append(f"신규 (<5회): {newcomers}  ")
        lines.append(f"평균 미만: {at_risk_count}")

        # Show top at-risk artists
        at_risk = pool_stats.get('at_risk', [])
        if at_risk:
            lines.append("")
            lines.append("**교체 가능성이 높은 작가:**")
            for artist, elo, matches, weight in at_risk[:5]:
                lines.append(f"- {artist} ({elo:.0f})")

        # Show recent pool changes
        if self.rotation_log:
            lines.append("")
            lines.append("---")
            lines.append("### 최근 풀 변경")
            lines.append("*최신순:*")
            for rot_type, artist, elo, extra in self.rotation_log[:8]:
                if rot_type == "in":
                    status = "[returning]" if extra else "[new]"
                    lines.append(f"+ **{artist}** ({elo:.0f}) {status}")
                else:
                    lines.append(f"- ~~{artist}~~ ({elo:.0f})")

        return "\n".join(lines)

    async def generate_new_comparison(
        self,
        custom_prompt: str,
        custom_negative_prompt: str = "",
        quality_toggle: bool = True,
        uc_preset: int = 0,
        ranking_mode: str = RANKING_MODE_BALANCED,
    ):
        """Generate a new pair of images for comparison."""
        custom_prompt = custom_prompt or ""
        custom_negative_prompt = custom_negative_prompt or ""
        self.artist_manager.set_ranking_mode(ranking_mode)

        # Use custom prompt if provided, otherwise default
        base_prompt = custom_prompt.strip() if custom_prompt.strip() else DEFAULT_PROMPT

        # Use custom negative prompt if provided, otherwise None (will use default)
        negative_prompt = custom_negative_prompt.strip() if custom_negative_prompt.strip() else None

        # Remove any existing artist tags from the prompt
        base_prompt = remove_artist_tags(base_prompt)

        # Ensure we have the artist placeholder or a clean prompt
        if "{artist_placeholder}" not in base_prompt:
            # Add placeholder if not present
            base_prompt = insert_artist_tags(base_prompt, "{artist_placeholder}")

        self.current_prompt = base_prompt
        self.current_custom_prompt = custom_prompt
        self.current_negative_prompt = custom_negative_prompt
        self.current_quality_toggle = bool(quality_toggle)
        self.current_uc_preset = int(uc_preset)

        try:
            session = self.get_session()
        except ValueError as e:
            # API key not configured
            error_msg = (
                "**NovelAI API 키가 설정되지 않았습니다.**\n\n"
                "배포 서비스의 비밀 환경변수에 `NOVELAI_API_KEY`를 등록한 뒤 "
                "앱을 다시 시작하세요. 토큰을 프롬프트나 채팅에 붙여 넣지 마세요."
            )
            return (
                None,
                None,
                error_msg,
                self.format_top_artists_display(),
                "",
                "",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

        path_a, path_b, artists_a, artists_b = await generate_comparison_pair(
            base_prompt,
            self.artist_manager,
            session,
            COMPARISON_IMAGES_DIR,
            negative_prompt,
            quality_toggle,
            uc_preset
        )

        if path_a and path_b:
            self.current_image_a = path_a
            self.current_image_b = path_b
            self.current_artists_a = artists_a
            self.current_artists_b = artists_b

            # Reset selection state for new comparison
            # BUT keep undo state so user can still undo the previous selection!
            self.selection_made = False
            # Don't clear last_undo_state here - it persists until next selection
            self.save_current_comparison()

            # Undo is available if we have a previous state to restore
            can_undo = self.last_undo_state is not None

            return (
                str(path_a),
                str(path_b),
                "이미지가 생성되었습니다. 더 마음에 드는 쪽을 선택하세요.",
                self.format_top_artists_display(),
                "",  # Clear result_msg
                "",  # Clear details_msg
                gr.update(interactive=True),   # Enable pick_a
                gr.update(interactive=True),   # Enable pick_b
                gr.update(interactive=can_undo),  # Undo available if we have state
            )
        else:
            return (
                None,
                None,
                "이미지 생성에 실패했습니다. 잠시 후 건너뛰기를 눌러 다시 시도하세요.",
                self.format_top_artists_display(),
                "",
                "",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

    def pick_winner(self, winner: str):
        """Process a winner selection. Returns tuple for UI update."""
        if not self.current_artists_a or not self.current_artists_b:
            return (
                "진행 중인 비교가 없습니다. 먼저 이미지를 생성하세요.",
                "",
                self.format_top_artists_display(),
                gr.update(interactive=True),  # pick_a
                gr.update(interactive=True),  # pick_b
                gr.update(interactive=False),  # undo
            )

        if self.selection_made:
            return (
                "이미 선택한 비교입니다. 되돌리거나 다음 이미지를 생성하세요.",
                "",
                self.format_top_artists_display(),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=True),
            )

        if winner == "A":
            winners = self.current_artists_a
            losers = self.current_artists_b
        else:
            winners = self.current_artists_b
            losers = self.current_artists_a

        # Save state for undo BEFORE making changes
        old_ratings = {a: self.elo_system.get_rating(a) for a in winners + losers}
        old_comparisons = {a: self.elo_system.get_artist_comparison_count(a) for a in winners + losers}

        # Update ELO ratings
        self.elo_system.update_ratings(winners, losers)
        self.elo_system.save(ELO_RATINGS_FILE)

        # Update active pool (rotate losers, introduce new artists)
        rotated_out, rotated_in = self.artist_manager.process_result(winners, losers)

        # Log rotations (most recent first)
        for artist, elo, is_returning in rotated_in:
            self.rotation_log.insert(0, ("in", artist, elo, is_returning))
        for artist, elo in rotated_out:
            self.rotation_log.insert(0, ("out", artist, elo, None))
        # Keep only last 20 entries
        self.rotation_log = self.rotation_log[:20]

        # Save undo state (including current images for restoration)
        # Extract just artist names for undo
        rotated_out_names = [artist for artist, elo in rotated_out]
        rotated_in_names = [artist for artist, elo, is_returning in rotated_in]
        self.last_undo_state = UndoState(
            winners=winners.copy(),
            losers=losers.copy(),
            old_ratings=old_ratings,
            old_comparisons=old_comparisons,
            rotated_out=rotated_out_names,
            rotated_in=rotated_in_names,
            prev_image_a=str(self.current_image_a) if self.current_image_a else None,
            prev_image_b=str(self.current_image_b) if self.current_image_b else None,
            prev_artists_a=self.current_artists_a.copy(),
            prev_artists_b=self.current_artists_b.copy(),
        )
        self.selection_made = True
        self.save_current_comparison()

        # Record history
        record = ComparisonRecord(
            timestamp=time.time(),
            artists_a=self.current_artists_a,
            artists_b=self.current_artists_b,
            winner=winner,
            image_a_path=str(self.current_image_a) if self.current_image_a else "",
            image_b_path=str(self.current_image_b) if self.current_image_b else ""
        )
        self.history.add_record(record)

        # Format result message
        winner_artists = ", ".join(winners)
        loser_artists = ", ".join(losers)
        result_msg = f"**승리:** {winner_artists}\n**패배:** {loser_artists}"

        # Show artist details
        details = "### 작가 상세\n"
        details += f"**이미지 A:** {', '.join(self.current_artists_a)}\n"
        details += f"**이미지 B:** {', '.join(self.current_artists_b)}\n\n"
        details += "**변경된 ELO:**\n"
        for artist in winners + losers:
            rating = self.elo_system.get_rating(artist)
            details += f"- {artist}: {rating:.0f}\n"

        return (
            result_msg,
            details,
            self.format_top_artists_display(),
            gr.update(interactive=False),  # Disable pick_a
            gr.update(interactive=False),  # Disable pick_b
            gr.update(interactive=True),   # Enable undo
        )

    def undo_last_selection(self):
        """Undo the last selection and restore previous images."""
        if not self.last_undo_state:
            return (
                None,  # image_a
                None,  # image_b
                "되돌릴 선택이 없습니다.",
                self.format_top_artists_display(),
                "",
                "",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

        state = self.last_undo_state

        # Restore old ratings
        for artist, old_rating in state.old_ratings.items():
            self.elo_system.ratings[artist] = old_rating

        # Restore old comparison counts
        for artist, old_count in state.old_comparisons.items():
            self.elo_system.artist_comparisons[artist] = old_count

        # Decrement total comparison count
        self.elo_system.comparison_count -= 1
        self.elo_system.save(ELO_RATINGS_FILE)

        # Restore all active-pool changes, including newly added artists.
        if state.rotated_out or state.rotated_in:
            self.artist_manager.revert_rotation(state.rotated_out, state.rotated_in)

        # Remove the corresponding entries from the in-memory rotation log.
        for rotation_type, artists in (
            ("out", state.rotated_out),
            ("in", state.rotated_in),
        ):
            for artist in artists:
                for index, entry in enumerate(self.rotation_log):
                    if entry[0] == rotation_type and entry[1] == artist:
                        self.rotation_log.pop(index)
                        break

        # Remove last history record
        if self.history.records:
            self.history.records.pop()
            self.history.save()

        # Restore previous images and artists
        self.current_image_a = Path(state.prev_image_a) if state.prev_image_a else None
        self.current_image_b = Path(state.prev_image_b) if state.prev_image_b else None
        self.current_artists_a = state.prev_artists_a.copy()
        self.current_artists_b = state.prev_artists_b.copy()

        # Clear undo state and reset selection
        self.last_undo_state = None
        self.selection_made = False
        self.save_current_comparison()

        return (
            state.prev_image_a,  # image_a
            state.prev_image_b,  # image_b
            "**선택을 되돌렸습니다.** 다시 선택하세요.",
            self.format_top_artists_display(),
            "",  # Clear result_msg
            "",  # Clear details_msg
            gr.update(interactive=True),   # Enable pick_a
            gr.update(interactive=True),   # Enable pick_b
            gr.update(interactive=False),  # Disable undo
        )

    def create_ui(self) -> gr.Blocks:
        """Create the Gradio interface."""

        with gr.Blocks(title="Artist ELO Ranker") as app:
            with gr.Column(elem_id="app-header"):
                gr.Markdown("# Artist ELO Ranker")
                gr.Markdown(
                    "두 이미지를 비교해 더 마음에 드는 쪽을 선택하세요. "
                    "선택 전에는 작가 태그가 숨겨집니다.  \n"
                    "<span id='desktop-help'>키보드: `1` A 선택 · `2` B 선택 · `S` 건너뛰기 · `0` 되돌리기</span>"
                )

            with gr.Row(elem_id="main-layout"):
                # Main comparison area (left side)
                with gr.Column(scale=2):
                    ranking_mode_dropdown = gr.Dropdown(
                        label="평가 방향",
                        choices=RANKING_MODE_CHOICES,
                        value=self.artist_manager.get_ranking_mode(),
                        interactive=True,
                    )
                    ranking_mode_help = gr.Markdown(
                        get_ranking_mode_description(
                            self.artist_manager.get_ranking_mode()
                        )
                    )

                    # Prompt input
                    with gr.Accordion("생성 설정", open=False):
                        prompt_input = gr.Textbox(
                            label="프롬프트",
                            placeholder="비워 두면 기본 프롬프트를 사용합니다. 작가 태그는 자동으로 들어갑니다.",
                            lines=4,
                            value=self.current_custom_prompt,
                        )
                        negative_prompt_input = gr.Textbox(
                            label="네거티브 프롬프트",
                            placeholder="비워 두면 기본 네거티브 프롬프트를 사용합니다.",
                            lines=3,
                            value=self.current_negative_prompt,
                        )
                        with gr.Row():
                            quality_toggle = gr.Checkbox(
                                label="품질 태그 추가",
                                value=self.current_quality_toggle,
                                info="very aesthetic, masterpiece, no text를 추가합니다.",
                            )
                            uc_preset_dropdown = gr.Dropdown(
                                label="자동 네거티브 프리셋",
                                choices=[
                                    ("없음", -1),
                                    ("Heavy · 표준 품질 필터", 0),
                                    ("Light · 최소 필터", 1),
                                    ("Human Focus · 인물 중심", 2),
                                    ("Heavy + Anatomy · 신체 보정", 3),
                                ],
                                value=self.current_uc_preset,
                                interactive=True,
                            )
                        gr.Markdown(
                            "*`{artist_placeholder}`를 넣으면 해당 위치에 작가 태그가 삽입됩니다.*"
                        )

                    # Status message
                    status_msg = gr.Markdown("비교 이미지를 준비하고 있습니다…", elem_id="status-card")

                    # Image comparison
                    with gr.Row(elem_id="comparison-row"):
                        with gr.Column(elem_classes=["image-card"]):
                            image_a = gr.Image(
                                label="이미지 A",
                                type="filepath",
                                buttons=["fullscreen"],
                                elem_classes=["comparison-image"],
                            )
                            artists_a_display = gr.Markdown("", visible=False)

                        with gr.Column(elem_classes=["image-card"]):
                            image_b = gr.Image(
                                label="이미지 B",
                                type="filepath",
                                buttons=["fullscreen"],
                                elem_classes=["comparison-image"],
                            )
                            artists_b_display = gr.Markdown("", visible=False)

                    # Large, thumb-friendly controls stay reachable while scrolling.
                    with gr.Row(elem_id="vote-dock"):
                        pick_a_btn = gr.Button("A 선택", variant="primary", size="lg", elem_id="vote-a")
                        pick_b_btn = gr.Button("B 선택", variant="primary", size="lg", elem_id="vote-b")

                    with gr.Row(elem_id="secondary-actions"):
                        skip_btn = gr.Button("건너뛰기", variant="secondary", size="sm", elem_id="skip-button")
                        undo_btn = gr.Button(
                            "마지막 선택 되돌리기",
                            variant="stop",
                            size="sm",
                            interactive=False,
                            elem_id="undo-button",
                        )
                    show_artists_toggle = gr.Checkbox(label="작가 태그 보기", value=False)

                    # Result display
                    result_msg = gr.Markdown("")
                    details_msg = gr.Markdown("")

                # Leaderboard (right side)
                with gr.Column(scale=1):
                    with gr.Accordion("랭킹과 풀 통계", open=True):
                        leaderboard = gr.Markdown(
                            self.format_top_artists_display(),
                            label="Top Artists",
                        )
                    export_btn = gr.Button("랭킹 CSV 내보내기")
                    export_file = gr.File(label="Download", visible=False)

                    # Comparison history panel
                    with gr.Accordion("최근 비교 기록", open=False):
                        history_display = gr.Markdown(self.format_recent_history())

            # Event handlers
            def on_generate(prompt, negative_prompt, quality_tags, uc_preset, ranking_mode):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.generate_new_comparison(
                            prompt,
                            negative_prompt,
                            quality_tags,
                            uc_preset,
                            ranking_mode,
                        )
                    )
                    # Add artist display text and history to result
                    artists_a_text = f"**작가:** {', '.join(self.current_artists_a)}"
                    artists_b_text = f"**작가:** {', '.join(self.current_artists_b)}"
                    history_text = self.format_recent_history()
                    return result + (artists_a_text, artists_b_text, history_text)
                finally:
                    loop.close()

            def on_initial_load(prompt, negative_prompt, quality_tags, uc_preset, ranking_mode):
                """Reuse a saved pair on refresh; only generate when none exists."""
                self.artist_manager.set_ranking_mode(ranking_mode)
                has_saved_pair = (
                    self.current_image_a
                    and self.current_image_b
                    and self.current_image_a.is_file()
                    and self.current_image_b.is_file()
                    and self.current_artists_a
                    and self.current_artists_b
                )
                if not has_saved_pair:
                    return on_generate(
                        prompt,
                        negative_prompt,
                        quality_tags,
                        uc_preset,
                        ranking_mode,
                    )

                can_pick = not self.selection_made
                status = (
                    "저장된 비교를 불러왔습니다. 더 마음에 드는 이미지를 선택하세요."
                    if can_pick
                    else "이 비교는 이미 선택되었습니다. 건너뛰기를 눌러 다음 비교를 생성하세요."
                )
                return (
                    str(self.current_image_a),
                    str(self.current_image_b),
                    status,
                    self.format_top_artists_display(),
                    "",
                    "",
                    gr.update(interactive=can_pick),
                    gr.update(interactive=can_pick),
                    gr.update(interactive=self.last_undo_state is not None),
                    f"**작가:** {', '.join(self.current_artists_a)}",
                    f"**작가:** {', '.join(self.current_artists_b)}",
                    self.format_recent_history(),
                )

            def on_pick_then_generate(prompt, negative_prompt, quality_tags, uc_preset, ranking_mode):
                """Generate new comparison after pick (for chaining)."""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.generate_new_comparison(
                            prompt,
                            negative_prompt,
                            quality_tags,
                            uc_preset,
                            ranking_mode,
                        )
                    )
                    artists_a_text = f"**작가:** {', '.join(self.current_artists_a)}"
                    artists_b_text = f"**작가:** {', '.join(self.current_artists_b)}"
                    history_text = self.format_recent_history()
                    return result + (artists_a_text, artists_b_text, history_text)
                finally:
                    loop.close()

            def on_export():
                """Export leaderboard as downloadable CSV file."""
                content = self.export_leaderboard_csv()
                filepath = COMPARISON_IMAGES_DIR / "leaderboard_export.csv"
                filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                return str(filepath)

            def on_toggle_artists(show):
                """Toggle visibility of artist tags."""
                return (
                    gr.update(visible=show),
                    gr.update(visible=show),
                )

            def on_undo():
                """Undo and restore artist display text."""
                result = self.undo_last_selection()
                artists_a_text = f"**작가:** {', '.join(self.current_artists_a)}" if self.current_artists_a else ""
                artists_b_text = f"**작가:** {', '.join(self.current_artists_b)}" if self.current_artists_b else ""
                history_text = self.format_recent_history()
                return result + (artists_a_text, artists_b_text, history_text)

            def on_mode_change(ranking_mode):
                """Persist the preset without generating or resetting any ranking data."""
                active_mode = self.artist_manager.set_ranking_mode(ranking_mode)
                return (
                    get_ranking_mode_description(active_mode),
                    self.format_top_artists_display(),
                )

            ranking_mode_dropdown.change(
                fn=on_mode_change,
                inputs=[ranking_mode_dropdown],
                outputs=[ranking_mode_help, leaderboard],
            )

            # Auto-generate first comparison on app load
            app.load(
                fn=on_initial_load,
                inputs=[prompt_input, negative_prompt_input, quality_toggle, uc_preset_dropdown, ranking_mode_dropdown],
                outputs=[image_a, image_b, status_msg, leaderboard, result_msg, details_msg, pick_a_btn, pick_b_btn, undo_btn, artists_a_display, artists_b_display, history_display]
            )

            # Pick A: update ELO, then auto-generate next pair, then refresh CSV
            pick_a_btn.click(
                fn=lambda: self.pick_winner("A"),
                outputs=[result_msg, details_msg, leaderboard, pick_a_btn, pick_b_btn, undo_btn]
            ).then(
                fn=on_pick_then_generate,
                inputs=[prompt_input, negative_prompt_input, quality_toggle, uc_preset_dropdown, ranking_mode_dropdown],
                outputs=[image_a, image_b, status_msg, leaderboard, result_msg, details_msg, pick_a_btn, pick_b_btn, undo_btn, artists_a_display, artists_b_display, history_display]
            ).then(
                fn=on_export,
                outputs=[export_file]
            )

            # Pick B: update ELO, then auto-generate next pair, then refresh CSV
            pick_b_btn.click(
                fn=lambda: self.pick_winner("B"),
                outputs=[result_msg, details_msg, leaderboard, pick_a_btn, pick_b_btn, undo_btn]
            ).then(
                fn=on_pick_then_generate,
                inputs=[prompt_input, negative_prompt_input, quality_toggle, uc_preset_dropdown, ranking_mode_dropdown],
                outputs=[image_a, image_b, status_msg, leaderboard, result_msg, details_msg, pick_a_btn, pick_b_btn, undo_btn, artists_a_display, artists_b_display, history_display]
            ).then(
                fn=on_export,
                outputs=[export_file]
            )

            # Undo: restore previous state and images, then refresh CSV
            undo_btn.click(
                fn=on_undo,
                outputs=[image_a, image_b, status_msg, leaderboard, result_msg, details_msg, pick_a_btn, pick_b_btn, undo_btn, artists_a_display, artists_b_display, history_display]
            ).then(
                fn=on_export,
                outputs=[export_file]
            )

            # Toggle artist visibility
            show_artists_toggle.change(
                fn=on_toggle_artists,
                inputs=[show_artists_toggle],
                outputs=[artists_a_display, artists_b_display]
            )

            # Skip: generate new images without any ELO changes
            skip_btn.click(
                fn=on_pick_then_generate,
                inputs=[prompt_input, negative_prompt_input, quality_toggle, uc_preset_dropdown, ranking_mode_dropdown],
                outputs=[image_a, image_b, status_msg, leaderboard, result_msg, details_msg, pick_a_btn, pick_b_btn, undo_btn, artists_a_display, artists_b_display, history_display]
            )

            # Export leaderboard - generate CSV and show download link
            export_btn.click(
                fn=on_export,
                outputs=[export_file]
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=[export_file]
            )

            # Keyboard shortcuts via JavaScript
            app.load(
                fn=None,
                js="""
                () => {
                    document.addEventListener('keydown', (e) => {
                        // Ignore if typing in a text field
                        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

                        const clickControl = (id) => {
                            const root = document.getElementById(id);
                            if (!root) return false;
                            const button = root.matches('button') ? root : root.querySelector('button');
                            if (!button || button.disabled || button.getAttribute('aria-disabled') === 'true') return false;
                            button.click();
                            return true;
                        };

                        if (e.key === '1') {
                            clickControl('vote-a');
                        } else if (e.key === '2') {
                            clickControl('vote-b');
                        } else if (e.key === 's' || e.key === 'S') {
                            clickControl('skip-button');
                        } else if (e.key === '0') {
                            clickControl('undo-button');
                        }
                    });
                    return [];
                }
                """
            )

        return app


# --------------------------------------------------------------------------------
# Main Entry Point
# --------------------------------------------------------------------------------

def main():
    """Main entry point."""
    print("Starting Artist ELO Ranker...")
    print(f"Artist tags file: {ARTIST_TAGS_FILE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"ELO ratings file: {ELO_RATINGS_FILE}")
    print(f"Comparison images dir: {COMPARISON_IMAGES_DIR}")

    public_bind = SERVER_HOST not in {"127.0.0.1", "localhost", "::1"}
    if public_bind and len(APP_PASSWORD) < 8:
        raise ValueError(
            "APP_PASSWORD must contain at least 8 characters when SERVER_HOST "
            "is exposed beyond this device. Set it as a secret environment variable."
        )

    ranker = ArtistELORanker()
    app = ranker.create_ui()
    auth = (APP_USERNAME, APP_PASSWORD) if APP_PASSWORD else None

    # Launch the app
    app.launch(
        share=False,
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        inbrowser=INBROWSER,
        auth=auth,
        auth_message="개인용 Artist ELO입니다. 설정한 계정으로 로그인하세요.",
        theme=gr.themes.Soft(),
        css=MOBILE_CSS,
        head=APP_HEAD,
        pwa=True,
        allowed_paths=[str(COMPARISON_IMAGES_DIR)],
        enable_monitoring=False,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
