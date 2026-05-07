# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for ``balance.interop.diff_diff`` and its shared helpers.

The test classes split into two groups:

* Package-agnostic helpers (rows T1-T7, T15-T17): apply to ANY future
  ``balance.interop.*`` adapter -- e.g. ``ActiveWeightColumnTest``,
  ``DropHistoryColumnsTest``, ``ConventionsTest``,
  ``AttachBalanceProvenanceTest``, ``ValidateRowCountTest``.
* diff-diff-specific (rows T8-T14, T18-T20): apply only to
  ``balance.interop.diff_diff`` -- e.g. ``ToSurveyDesignTest``,
  ``ToPanelForDidTest``, ``FitDidTest``, ``AsBalanceDiagnosticTest``.

Each test docstring tags its row(s). The :class:`ImportGuardTest` is
the only test class that always runs even without diff-diff installed.
"""

from __future__ import annotations

import unittest
import warnings
from typing import Any

import balance
import numpy as np
import pandas as pd

try:
    import diff_diff as _dd  # noqa: F401

    _DIFF_DIFF_AVAILABLE: bool = True
except ImportError:
    _DIFF_DIFF_AVAILABLE = False


def _toy_balanced_panel() -> pd.DataFrame:
    """Tiny 8-unit / 3-period panel sufficient for adapter wiring tests.

    Multi-period staggered design: half the units are never-treated
    (``first_treat=0``) and half are treated at ``t=2``. This shape is the
    natural input for diff-diff's staggered estimators (``CallawaySantAnna``,
    ``StackedDiD``, etc.) which consume the cohort-year column directly via
    ``treatment_first=`` rather than a binary post-treatment indicator. The
    simpler 2x2 ``DifferenceInDifferences`` estimator is a poor match here:
    it would require both ``time`` and ``treatment`` to be binary 0/1
    vectors (see ``diff_diff.utils.validate_binary``).

    The survey weight ``w`` is drawn ONCE per unit and repeated across all
    of its time periods. Diff-diff's panel estimators
    (``CallawaySantAnna``, ``ContinuousDiD``, ``EfficientDiD``) collapse
    panel-level survey design columns down to one row per unit and require
    those columns to be constant within unit
    (``diff_diff.survey._validate_unit_constant_survey``). A per-row
    weight would fail that check before the test ever exercised the
    adapter's provenance behaviour.
    """
    rng: np.random.Generator = np.random.default_rng(20260430)
    rows: list[dict[str, float]] = []
    row_id: int = 0
    for unit in range(8):
        first_treat: int = 0 if unit < 4 else 2
        # Unit-constant survey weight: realistic for sampling weights too.
        unit_weight: float = float(rng.uniform(0.5, 2.0))
        for t in range(3):
            post_treatment: bool = first_treat > 0 and t >= first_treat
            rows.append(
                {
                    "id": float(row_id),
                    "unit": float(unit),
                    "t": float(t),
                    "first_treat": float(first_treat),
                    "x1": float(rng.normal()),
                    "x2": float(rng.normal()),
                    "y": float(rng.normal()) + (1.0 if post_treatment else 0.0),
                    "w": unit_weight,
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


def _toy_target() -> pd.DataFrame:
    rng: np.random.Generator = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "unit": np.arange(50, dtype=float),
            "x1": rng.normal(size=50),
            "x2": rng.normal(size=50),
            "w": np.ones(50),
        }
    )


def _make_sample() -> "balance.Sample":
    """Build a balance.Sample on the toy panel for reuse across tests."""
    return balance.Sample.from_frame(
        _toy_balanced_panel(),
        id_column="id",
        weight_column="w",
        outcome_columns=["y"],
    )


# ---------------------------------------------------------------------------
# Helper-level tests (run regardless of diff-diff availability)
# ---------------------------------------------------------------------------


class ActiveWeightColumnTest(unittest.TestCase):
    """§11.A — ``active_weight_column`` resolves the active weight name."""

    def test_returns_weight_column_name_when_set(self) -> None:
        """T1 — adapter forwards the column NAME, not the Series."""
        from balance.interop._common import active_weight_column

        s = _make_sample()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.assertEqual("w", active_weight_column(s))

    def test_raises_when_no_weight_column(self) -> None:
        """T2 — ``Sample`` without an active weight raises ``ValueError``."""
        from balance.interop._common import active_weight_column

        df: pd.DataFrame = _toy_balanced_panel().drop(columns=["w"])
        s = balance.Sample.from_frame(df, id_column="id", outcome_columns=["y"])
        # Force weight_column to None to simulate user-constructed Sample
        # with no weights.
        # pyre-ignore[16]: test-only access to private state.
        s._weight_column_name = None
        with self.assertRaises(ValueError):
            active_weight_column(s)


class DropHistoryColumnsTest(unittest.TestCase):
    """§11.A — ``drop_history_columns`` strips balance bookkeeping cols."""

    def test_drops_pre_adjust_and_adjusted_n(self) -> None:
        """T4 — ``weight_pre_adjust`` and ``weight_adjusted_*`` are stripped."""
        from balance.interop._common import drop_history_columns

        df: pd.DataFrame = _toy_balanced_panel()
        df["weight_pre_adjust"] = 1.0
        df["weight_adjusted_1"] = 1.1
        df["weight_adjusted_2"] = 1.2
        cleaned: pd.DataFrame = drop_history_columns(df)
        self.assertNotIn("weight_pre_adjust", cleaned.columns)
        self.assertNotIn("weight_adjusted_1", cleaned.columns)
        self.assertNotIn("weight_adjusted_2", cleaned.columns)
        # Sanity: the active weight column is NOT history-bookkeeping.
        self.assertIn("w", cleaned.columns)

    def test_drops_weight_trimmed_columns(self) -> None:
        """T4 (extension) — ``weight_trimmed_*`` is also stripped."""
        from balance.interop._common import drop_history_columns

        df: pd.DataFrame = _toy_balanced_panel()
        df["weight_trimmed_1"] = 1.0
        cleaned: pd.DataFrame = drop_history_columns(df)
        self.assertNotIn("weight_trimmed_1", cleaned.columns)

    def test_no_history_columns_passthrough(self) -> None:
        """T4 (extension) — frames with no bookkeeping cols pass through."""
        from balance.interop._common import drop_history_columns

        df: pd.DataFrame = _toy_balanced_panel()
        cleaned: pd.DataFrame = drop_history_columns(df)
        self.assertEqual(set(df.columns), set(cleaned.columns))


class ConventionsTest(unittest.TestCase):
    """§11.A — column-name convention constants are stable."""

    def test_weight_column_constant(self) -> None:
        """Default weight column is ``"weight"`` (matches balance's
        post-``adjust()`` active column name)."""
        from balance.interop import conventions

        self.assertEqual("weight", conventions.WEIGHT_COLUMN)

    def test_default_design_columns_includes_weights(self) -> None:
        """``DEFAULT_DESIGN_COLUMNS`` exposes a ``"weights"`` key whose
        value is the canonical ``WEIGHT_COLUMN`` constant."""
        from balance.interop import conventions

        self.assertEqual(
            conventions.WEIGHT_COLUMN,
            conventions.DEFAULT_DESIGN_COLUMNS["weights"],
        )

    def test_weight_type_default_is_pweight(self) -> None:
        """``WEIGHT_TYPE_DEFAULT`` is ``"pweight"`` — required for the
        full diff-diff estimator family."""
        from balance.interop import conventions

        self.assertEqual("pweight", conventions.WEIGHT_TYPE_DEFAULT)


class AttachBalanceProvenanceTest(unittest.TestCase):
    """§11.A — ``attach_balance_provenance`` writes ``_balance_adjustment``."""

    def test_attaches_to_plain_object(self) -> None:
        """T15 — provenance attribute lands on a vanilla object."""
        from balance.interop._common import attach_balance_provenance

        class _Stub:
            pass

        target = _Stub()
        sample = _make_sample()
        attach_balance_provenance(target, sample)
        # ``_balance_adjustment`` is dynamically set by
        # ``attach_balance_provenance``; use ``getattr`` so Pyre does not
        # require a declared attribute on ``_Stub`` (which would defeat
        # the point of testing the side-channel attach behavior — the
        # ``hasattr`` early-return path inside the helper would skip the
        # assignment if a class-level attribute were declared).
        self.assertIs(sample, getattr(target, "_balance_adjustment", None))

    def test_idempotent_on_second_call(self) -> None:
        """T15 — second call does NOT clobber the original lineage."""
        from balance.interop._common import attach_balance_provenance

        class _Stub:
            pass

        target = _Stub()
        sample_a = _make_sample()
        sample_b = _make_sample()
        attach_balance_provenance(target, sample_a)
        attach_balance_provenance(target, sample_b)
        self.assertIs(sample_a, getattr(target, "_balance_adjustment", None))


class ValidateRowCountTest(unittest.TestCase):
    """§11.A — row-count guard catches ``na_action='drop'`` regressions."""

    def test_passes_when_counts_match(self) -> None:
        """T17 — equal counts: silent return (None)."""
        from balance.interop._common import validate_row_count

        s = _make_sample()
        # The function is documented to return ``None`` on a successful match
        # and to raise ``ValueError`` on mismatch (see _common.py). The
        # explicit assertion documents the success contract for the reader.
        self.assertIsNone(validate_row_count(s, len(s.df), ctx="test"))

    def test_raises_when_counts_differ(self) -> None:
        """T17 — mismatched counts surface a ``ValueError`` with ctx."""
        from balance.interop._common import validate_row_count

        s = _make_sample()
        with self.assertRaises(ValueError) as cm:
            validate_row_count(s, len(s.df) + 5, ctx="my_adapter")
        self.assertIn("my_adapter", str(cm.exception))


# ---------------------------------------------------------------------------
# diff-diff-specific tests (skipped when diff-diff isn't installed)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_DIFF_DIFF_AVAILABLE, "diff-diff is not installed")
class ToSurveyDesignTest(unittest.TestCase):
    """§11.B — ``to_survey_design`` builds a ``diff_diff.SurveyDesign``."""

    def test_uses_weight_column_name(self) -> None:
        """T1 — adapter forwards the column NAME, not the Series."""
        from balance.interop import diff_diff as bd

        s = _make_sample()
        d: Any = bd.to_survey_design(s)
        # SurveyDesign.weights is the column name string.
        self.assertEqual("w", d.weights)

    def test_default_weight_type_is_pweight(self) -> None:
        """T3 — adapter defaults to ``weight_type='pweight'``."""
        from balance.interop import diff_diff as bd

        s = _make_sample()
        d: Any = bd.to_survey_design(s)
        self.assertEqual("pweight", d.weight_type)

    def test_design_columns_rejects_weights_key(self) -> None:
        """T8/T9 — ``design_columns`` cannot reset ``"weights"``."""
        from balance.interop import diff_diff as bd

        s = _make_sample()
        with self.assertRaises(ValueError) as cm:
            bd.to_survey_design(s, design_columns={"weights": "w"})
        self.assertIn("design_columns cannot set 'weights'", str(cm.exception))

    def test_design_columns_rejects_unknown_keys(self) -> None:
        """T13 — unknown SurveyDesign fields raise ``ValueError``."""
        from balance.interop import diff_diff as bd

        s = _make_sample()
        with self.assertRaises(ValueError) as cm:
            bd.to_survey_design(s, design_columns={"not_a_real_field": "anything"})
        self.assertIn("unknown SurveyDesign fields", str(cm.exception))


@unittest.skipUnless(_DIFF_DIFF_AVAILABLE, "diff-diff is not installed")
class ToPanelForDidTest(unittest.TestCase):
    """§11.B — ``to_panel_for_did`` wraps ``aggregate_survey``."""

    def test_returns_two_tuple(self) -> None:
        """T11 — returns ``(panel_df, second_stage_design)``."""
        from balance.interop import diff_diff as bd

        s = _make_sample()
        # NOTE: only ``ImportError`` skips. Earlier revisions also swallowed
        # ``KeyError``, but a ``KeyError`` from ``aggregate_survey`` is a
        # genuine adapter contract regression (e.g. the adapter renamed
        # a column the upstream call still expects), not an "untestable
        # environment" signal — the adapter is the runtime contract with
        # diff-diff; we want CI to surface those, not skip them. We only
        # skip on ``ImportError`` to handle the rare build variant where
        # an optional cython sub-dependency of diff-diff is missing.
        # ``AssertionError``, ``AttributeError``, ``TypeError``, ``KeyError``
        # all fail the test.
        try:
            panel_df, second_stage = bd.to_panel_for_did(
                s, by=["unit", "t"], outcomes="y"
            )
        except ImportError as e:
            self.skipTest(f"aggregate_survey unavailable in this build: {e}")
        self.assertIsInstance(panel_df, pd.DataFrame)
        # Second-stage weight column auto-named ``"{first_outcome}_weight"``.
        self.assertIn(second_stage.weights, panel_df.columns)

    def test_default_second_stage_weight_type_is_pweight(self) -> None:
        """T11 — ``second_stage_weights`` defaults to ``"pweight"``."""
        from balance.interop import diff_diff as bd

        s = _make_sample()
        # See note above re: narrow exception list (ImportError only).
        try:
            _, second_stage = bd.to_panel_for_did(s, by=["unit", "t"], outcomes="y")
        except ImportError as e:
            self.skipTest(f"aggregate_survey unavailable in this build: {e}")
        self.assertEqual("pweight", second_stage.weight_type)

    def test_panel_weights_match_direct_aggregate_survey(self) -> None:
        """T-NEW (numerical roundtrip) — panel weights produced by the adapter
        match those produced by a direct ``aggregate_survey`` call on the
        same toy sample. The adapter is a thin wrapper; this test guards
        against refactors that would silently change weight aggregation
        semantics (e.g. swapping which column becomes the active weight,
        forgetting to forward ``second_stage_weights``)."""
        import diff_diff as dd
        from balance.interop import diff_diff as bd

        s = _make_sample()
        try:
            adapter_panel, adapter_design = bd.to_panel_for_did(
                s, by=["unit", "t"], outcomes="y"
            )
            # Direct route: replicate what to_panel_for_did does internally
            # (build first-stage SurveyDesign from s.weight_column, forward
            # the same second_stage_weights value, no history columns to
            # drop on this fixture).
            direct_design: Any = dd.SurveyDesign(
                weights="w", weight_type="pweight", lonely_psu="adjust"
            )
            direct_panel: pd.DataFrame
            direct_panel, _ = dd.aggregate_survey(
                s.df,
                by=["unit", "t"],
                outcomes="y",
                survey_design=direct_design,
                second_stage_weights="pweight",
            )
        except ImportError as e:
            self.skipTest(f"aggregate_survey unavailable in this build: {e}")

        # Both routes must produce the same row count and the same
        # auto-generated weight column name (``{first_outcome}_weight``).
        self.assertEqual(len(adapter_panel), len(direct_panel))
        weight_col: str = adapter_design.weights
        self.assertIn(weight_col, direct_panel.columns)

        # Element-wise numerical agreement on the second-stage weight column,
        # after sorting by the join keys to remove ordering noise.
        adapter_sorted: pd.DataFrame = adapter_panel.sort_values(
            ["unit", "t"]
        ).reset_index(drop=True)
        direct_sorted: pd.DataFrame = direct_panel.sort_values(
            ["unit", "t"]
        ).reset_index(drop=True)
        np.testing.assert_allclose(
            adapter_sorted[weight_col].to_numpy(),
            direct_sorted[weight_col].to_numpy(),
            rtol=1e-12,
            atol=0.0,
        )


class ValidateNonzeroWeightsTest(unittest.TestCase):
    """§11.A — ``validate_nonzero_weights`` rejects unusable weight vectors."""

    def test_passes_with_some_positive(self) -> None:
        """T-NEW — vector with at least one positive weight passes silently."""
        from balance.interop._common import validate_nonzero_weights

        df: pd.DataFrame = pd.DataFrame({"w": [0.0, 0.0, 0.5, 1.5]})
        # No exception, no return value -- mirrors validate_row_count contract.
        self.assertIsNone(validate_nonzero_weights(df, "w", ctx="my_adapter"))

    def test_raises_on_all_zero(self) -> None:
        """T-NEW — all-zero weights raise with an actionable message."""
        from balance.interop._common import validate_nonzero_weights

        df: pd.DataFrame = pd.DataFrame({"w": [0.0, 0.0, 0.0]})
        with self.assertRaises(ValueError) as cm:
            validate_nonzero_weights(df, "w", ctx="my_adapter")
        self.assertIn("my_adapter", str(cm.exception))
        self.assertIn("no positive entries", str(cm.exception))

    def test_raises_on_all_negative(self) -> None:
        """T-NEW — all-negative weights also raise (sum of squares is positive
        but the SurveyDesign still produces a meaningless fit)."""
        from balance.interop._common import validate_nonzero_weights

        df: pd.DataFrame = pd.DataFrame({"w": [-1.0, -2.0, -0.5]})
        with self.assertRaises(ValueError):
            validate_nonzero_weights(df, "w", ctx="my_adapter")

    def test_raises_on_mixed_sign_weights(self) -> None:
        """T-NEW (Copilot) — mixed-sign weights (e.g. ``[-1, 2]``) must also
        raise. The earlier guard checked only ``not (weights > 0).any()``,
        which was satisfied by the positive entry and let mixed-sign
        vectors through silently. Design weights are non-negative by
        contract (a negative entry usually points at the wrong column —
        a residual / centred / mean-deviation vector — not a design
        weight), so we now reject as soon as we see any negative entry
        before checking the positive-entry condition."""
        from balance.interop._common import validate_nonzero_weights

        df: pd.DataFrame = pd.DataFrame({"w": [-1.0, 2.0]})
        with self.assertRaises(ValueError) as cm:
            validate_nonzero_weights(df, "w", ctx="my_adapter")
        self.assertIn("negative", str(cm.exception))

    def test_raises_on_missing_column(self) -> None:
        """T-NEW — missing weight column fails fast with the same wording as
        ``drop_nan_weight_rows``."""
        from balance.interop._common import validate_nonzero_weights

        df: pd.DataFrame = pd.DataFrame({"x": [1.0, 2.0]})
        with self.assertRaises(ValueError) as cm:
            validate_nonzero_weights(df, "w", ctx="my_adapter")
        self.assertIn("not present in df.columns", str(cm.exception))


@unittest.skipUnless(_DIFF_DIFF_AVAILABLE, "diff-diff is not installed")
class FitDidTest(unittest.TestCase):
    """§11.B — ``fit_did`` resolves estimator and attaches provenance."""

    def test_attaches_balance_adjustment(self) -> None:
        """T15 — result carries ``_balance_adjustment`` for provenance.

        The attachment is the headline guarantee of ``preserve_adjustment``
        (the default for ``fit_did``), so this test enforces that exactly
        one of two things happens:

        1. ``_balance_adjustment`` is set on the result and points at the
           original ``Sample`` (the happy path for any result type that
           accepts ``setattr`` or has a writable ``__dict__``), OR
        2. ``attach_balance_provenance`` emits a ``UserWarning`` because
           the result type rejected both attachment paths (frozen
           dataclass / msgspec struct / read-only ``mappingproxy``).

        The previous formulation accepted ``None`` silently, which made the
        regression signal disappear when balance lineage stopped attaching
        on a real diff-diff result type.
        """
        from balance.interop import diff_diff as bd

        s = _make_sample()
        # See narrow-exception note in ToPanelForDidTest above. Real
        # AssertionError / TypeError / AttributeError from the estimator
        # should fail the test rather than silently skipping it -- the
        # whole point of this test is to catch regressions in the adapter
        # surface against diff-diff's runtime contract.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                # Use the staggered ``CallawaySantAnna`` estimator rather
                # than ``DifferenceInDifferences``: the simpler 2x2 DiD
                # estimator validates BOTH ``time`` and ``treatment`` as
                # binary 0/1 vectors (see
                # ``diff_diff.utils.validate_binary``), which is
                # incompatible with this multi-period staggered fixture.
                # Callaway-Sant'Anna is the natural fit for the ``first_treat``
                # cohort encoding the fixture already provides, and it
                # exercises the same provenance-attachment code path.
                res: object = bd.fit_did(
                    s,
                    estimator="CallawaySantAnna",
                    outcome="y",
                    time="t",
                    unit="unit",
                    treatment_first="first_treat",
                )
            except ImportError as e:
                # See ToPanelForDidTest comment — only ImportError is treated
                # as untestable-environment. KeyError / AssertionError / etc.
                # propagate so CI catches real adapter-contract regressions.
                self.skipTest(f"DiD fit unavailable in this build: {e}")
        attached: bool = getattr(res, "_balance_adjustment", None) is s
        provenance_warned: bool = any(
            issubclass(w.category, UserWarning)
            and "_balance_adjustment" in str(w.message)
            for w in caught
        )
        self.assertTrue(
            attached or provenance_warned,
            msg=(
                "fit_did(preserve_adjustment=True) must either attach "
                "`_balance_adjustment` to the result or emit a UserWarning "
                "explaining why it could not. Got neither -- provenance "
                "is silently broken."
            ),
        )

    def test_rejects_unknown_estimator(self) -> None:
        """T19 — bad estimator name raises a clear ``ValueError``."""
        from balance.interop import diff_diff as bd

        s = _make_sample()
        with self.assertRaises(ValueError) as cm:
            bd.fit_did(
                s,
                estimator="NotARealEstimator",
                outcome="y",
                time="t",
                unit="unit",
                treatment_first="first_treat",
            )
        self.assertIn("NotARealEstimator", str(cm.exception))

    def test_rejects_reserved_positional_kwargs(self) -> None:
        """T-NEW — kwargs that collide with positional / wired-in fit args
        raise ``TypeError`` with an actionable message rather than the
        cryptic ``TypeError: multiple values for argument 'data'`` that the
        unguarded splat would produce.

        Limited to names that genuinely flow through ``**estimator_kwargs``:
        ``data`` / ``df`` (would collide with the positional ``df`` argument
        passed to ``instance.fit(df, ...)``) and ``first_treat`` (the
        diff-diff column-name kwarg; fit_did exposes the same column via
        the ``treatment_first=`` adapter param). Names already in fit_did's
        explicit signature (``outcome``, ``time``, ``unit``, ``treatment``,
        ``covariates``) bind to those params during the user's call and
        never reach ``**estimator_kwargs``; Python's own
        "multiple values for argument" ``TypeError`` covers same-name
        duplicates before fit_did's body runs, so an adapter-level guard
        for those names would be dead code -- and a test for them would
        either pass for the wrong reason (no collision at all) or trigger
        Python's TypeError instead of the adapter's, defeating the purpose
        of the assertion.
        """
        from balance.interop import diff_diff as bd

        s = _make_sample()
        for reserved_kwarg in ("data", "df", "first_treat"):
            with self.subTest(reserved_kwarg=reserved_kwarg):
                with self.assertRaises(TypeError) as cm:
                    bd.fit_did(
                        s,
                        estimator="CallawaySantAnna",
                        outcome="y",
                        time="t",
                        unit="unit",
                        treatment_first="first_treat",
                        **{reserved_kwarg: "anything"},  # type: ignore[arg-type]
                    )
                self.assertIn(reserved_kwarg, str(cm.exception))
                self.assertIn("**estimator_kwargs", str(cm.exception))

    def test_rejects_all_zero_weights(self) -> None:
        """T-NEW — a Sample whose active weight column is all-zero raises
        ``ValueError`` at the seam rather than producing an inf / NaN fit."""
        from balance.interop import diff_diff as bd

        df: pd.DataFrame = _toy_balanced_panel()
        df["w"] = 0.0  # zero out every weight
        s = balance.Sample.from_frame(
            df, id_column="id", weight_column="w", outcome_columns=["y"]
        )
        with self.assertRaises(ValueError) as cm:
            bd.fit_did(
                s,
                estimator="CallawaySantAnna",
                outcome="y",
                time="t",
                unit="unit",
                treatment_first="first_treat",
            )
        self.assertIn("no positive entries", str(cm.exception))


@unittest.skipUnless(_DIFF_DIFF_AVAILABLE, "diff-diff is not installed")
class AsBalanceDiagnosticTest(unittest.TestCase):
    """§11.B — ``as_balance_diagnostic`` returns a flat dict."""

    def test_returns_flat_dict_with_required_keys(self) -> None:
        """T15 (cont.) — required keys present, missing values are ``None``."""
        from balance.interop import diff_diff as bd

        s = _make_sample()

        class _StubResult:
            att: float = 0.5
            se: float = 0.1
            conf_int: tuple[float, float] = (0.3, 0.7)
            n_obs: int = 24
            survey_metadata: object | None = None

        diag: dict[str, object] = bd.as_balance_diagnostic(s, _StubResult())
        for key in (
            "att",
            "se",
            "conf_int",
            "n_obs",
            "balance_kish_ess",
            "balance_design_effect",
            "diff_diff_design_effect",
            "diff_diff_effective_n",
            "diff_diff_sum_weights",
            "balance_asmd_max_post",
            "balance_asmd_mean_post",
        ):
            self.assertIn(key, diag)
        self.assertEqual(0.5, diag["att"])
        self.assertIsNone(diag["diff_diff_design_effect"])

    def test_propagates_diff_diff_metadata(self) -> None:
        """``diff_diff_design_effect`` etc. propagate from
        ``did_results.survey_metadata`` when present."""
        from balance.interop import diff_diff as bd

        s = _make_sample()

        class _Meta:
            design_effect: float = 1.7
            effective_n: float = 200.0
            sum_weights: float = 250.0

        class _StubResult:
            att: float = 0.0
            survey_metadata: _Meta = _Meta()

        diag: dict[str, object] = bd.as_balance_diagnostic(s, _StubResult())
        self.assertEqual(1.7, diag["diff_diff_design_effect"])
        self.assertEqual(200.0, diag["diff_diff_effective_n"])
        self.assertEqual(250.0, diag["diff_diff_sum_weights"])


# ---------------------------------------------------------------------------
# Import guard (always runs)
# ---------------------------------------------------------------------------


class ImportGuardTest(unittest.TestCase):
    """The ONLY test class that runs in both modes (with and without
    diff-diff). When diff-diff is available, the module imports cleanly;
    when missing, calling into it raises a helpful ImportError pointing
    users at the ``balance[did]`` extra."""

    def test_module_importable(self) -> None:
        """Importing the submodule must always succeed (lazy guard).

        Even without diff-diff installed, the module must import cleanly
        and expose its public symbols — only calls into the adapter
        functions raise. We assert on the public symbols rather than just
        on the module to validate the import did something meaningful.
        """
        from balance.interop import diff_diff as bd

        self.assertIsNotNone(bd)
        for name in (
            "to_survey_design",
            "to_panel_for_did",
            "fit_did",
            "as_balance_diagnostic",
        ):
            self.assertTrue(
                hasattr(bd, name),
                f"balance.interop.diff_diff is missing public symbol {name!r}",
            )

    @unittest.skipIf(
        _DIFF_DIFF_AVAILABLE,
        "this assertion only fires when diff-diff is missing",
    )
    def test_calling_without_diff_diff_raises(self) -> None:
        """Calling adapter functions without diff-diff installed raises
        a helpful ``ImportError`` citing ``pip install balance[did]``."""
        from balance.interop import diff_diff as bd

        df: pd.DataFrame = pd.DataFrame(
            {"id": [0, 1, 2], "x": [1.0, 2.0, 3.0], "w": [1.0, 1.0, 1.0]}
        )
        s = balance.Sample.from_frame(df, id_column="id", weight_column="w")
        with self.assertRaises(ImportError) as cm:
            bd.to_survey_design(s)
        self.assertIn("balance[did]", str(cm.exception))
