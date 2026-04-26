"""
Abstract base class for feature builders.

Every feature module under ``src/features/`` exposes a subclass of
``FeatureBuilder`` that implements ``build()`` and returns a Polars DataFrame
keyed by ``SK_ID_CURR``. The ``assemble`` module joins all builders into the
three matrices described in PLAN §6.2.

Reference: PLAN_v2.md §2 (Feature Engineering Catalog) and §9 (output order).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl

from src import config
from src.utils import get_logger, timer

logger = get_logger()


class FeatureBuilder(ABC):
    """
    Base class for per-table feature engineering.

    Subclasses must:

    1. Set class attribute :attr:`name` (used for logging + parquet filename).
    2. Implement :meth:`build` to return a :class:`polars.DataFrame` with
       ``SK_ID_CURR`` as a key column and engineered features as the rest.

    Conventions
    -----------
    - Output column names are prefixed with the table abbreviation
      (``APP_``, ``BUR_``, ``PREV_``, ``INS_``, ``POS_``, ``CC_``) for traceability.
    - Builders never read raw CSVs directly — they consume processed parquet
      from ``data/processed/`` (written by :mod:`src.data`).
    - Builders are *idempotent* and *pure* given the input table.
    """

    #: Builder name. Override in subclasses (e.g. ``"application"``).
    name: str = "unnamed"

    #: Column-name prefix for outputs. Override in subclasses.
    prefix: str = ""

    def __init__(self) -> None:
        self.logger = logger

    @abstractmethod
    def build(self) -> pl.DataFrame:
        """
        Return a feature frame keyed by :data:`config.ID_COL` (``SK_ID_CURR``).

        Returns
        -------
        pl.DataFrame
            Columns: ``SK_ID_CURR`` + engineered features.
        """

    def run(self) -> pl.DataFrame:
        """
        Execute :meth:`build` with logging and basic shape assertions.
        """
        with timer(f"feature build: {self.name}"):
            df = self.build()
        self._assert_output(df)
        logger.info(
            f"  {self.name}: produced {df.width - 1} features over {df.height} rows"
        )
        return df

    def _assert_output(self, df: pl.DataFrame) -> None:
        """Validate the output frame contract."""
        if config.ID_COL not in df.columns:
            raise ValueError(
                f"{self.name}.build() must include {config.ID_COL!r} as a key column."
            )
        n_unique = df.n_unique(subset=[config.ID_COL])
        if n_unique != df.height:
            raise ValueError(
                f"{self.name}.build() output must have one row per {config.ID_COL}; "
                f"got {df.height} rows but {n_unique} unique IDs."
            )
        # Optional prefix discipline.
        if self.prefix:
            bad = [
                c
                for c in df.columns
                if c != config.ID_COL and not c.startswith(self.prefix)
            ]
            if bad:
                logger.warning(
                    f"  {self.name}: {len(bad)} columns lack expected prefix "
                    f"{self.prefix!r} (first 3: {bad[:3]})"
                )
