import unittest

from forecast_agents.tasks.panel import Bar, Panel, TaskSpec, synthetic_panel


class PanelTests(unittest.TestCase):
    def test_horizons_follow_sessions_and_labels_must_mature(self):
        days = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"]
        panel = Panel([Bar("A", day, 100 + i * 10) for i, day in enumerate(days)])
        self.assertAlmostEqual(panel.label("A", days[0], 1, days[1]), 0.1)
        self.assertAlmostEqual(panel.label("A", days[0], 3, days[3]), 0.3)
        self.assertAlmostEqual(panel.label("A", days[0], 5, days[5]), 0.5)
        with self.assertRaisesRegex(ValueError, "not matured"):
            panel.label("A", days[0], 5, days[4])

    def test_all_stocks_share_time_boundaries_and_purge_unmatured_origins(self):
        panel = synthetic_panel(3, 40)
        task = TaskSpec(panel.dates[15], panel.dates[29])
        validation = panel.origins(task, "validation")
        self.assertEqual(validation[0], panel.dates[16])
        self.assertEqual(validation[-1], panel.dates[24])
        test = panel.origins(task, "test")
        self.assertEqual(test[0], panel.dates[30])
        self.assertEqual(test[-1], panel.dates[34])
        self.assertTrue(all(row["date"] <= task.train_end for row in panel.history(task.train_end)))

    def test_walk_forward_refits_at_each_year_end_and_strides_within_folds(self):
        panel = synthetic_panel(1, 900)  # weekdays from 2020-01-01 to mid-2023
        task = TaskSpec("2020-12-31", "2022-12-30", refit="year", validation_stride=5)
        folds = panel.blocks(task, "validation")
        self.assertEqual([fold.fit_cutoff for fold in folds], ["2020-12-31", "2021-12-31"])
        self.assertTrue(all(day.startswith("2021") for day in folds[0].origins))
        self.assertTrue(all(day.startswith("2022") for day in folds[1].origins))
        for fold in folds:
            gaps = {panel.dates.index(b) - panel.dates.index(a) for a, b in zip(fold.origins, fold.origins[1:])}
            self.assertEqual(gaps, {5})
        # Validation labels mature by validation_end; nothing reaches into the test.
        self.assertLessEqual(panel.dates[panel.dates.index(folds[-1].origins[-1]) + 5], task.validation_end)
        test = panel.blocks(task, "test")
        self.assertEqual([block.fit_cutoff for block in test], ["2022-12-30"])
        self.assertEqual(test[0].origins[0], "2023-01-02")
        self.assertIn("2021-03-31", panel.period_ends("quarter"))

    def test_without_refits_one_model_fitted_at_train_end_serves_both_splits(self):
        panel = synthetic_panel(1, 40)
        task = TaskSpec(panel.dates[15], panel.dates[29])
        self.assertEqual([block.fit_cutoff for block in panel.blocks(task, "validation")], [task.train_end])
        self.assertEqual([block.fit_cutoff for block in panel.blocks(task, "test")], [task.train_end])
        self.assertEqual(task.protocol, "fixed_fit_at_train_end")
        with self.assertRaises(ValueError):
            TaskSpec(panel.dates[15], panel.dates[29], refit="weekly")

    def test_missing_sessions_are_rejected_instead_of_shortening_horizons(self):
        with self.assertRaisesRegex(ValueError, "balanced"):
            Panel([Bar("A", "2020-01-01", 1), Bar("A", "2020-01-02", 2),
                   Bar("B", "2020-01-01", 1)])

    def test_duplicate_dates_and_nonfinite_prices_are_rejected(self):
        for bars in ([Bar("A", "2020-01-01", 1)] * 2,
                     [Bar("A", "2020-01-01", float("nan"))]):
            with self.assertRaises(ValueError):
                Panel(bars)

    def test_synthetic_data_are_seeded(self):
        self.assertEqual(synthetic_panel(seed=1).prices, synthetic_panel(seed=1).prices)
        self.assertNotEqual(synthetic_panel(seed=1).prices, synthetic_panel(seed=2).prices)

    def test_noncanonical_dates_boolean_prices_and_backward_horizons_fail(self):
        for bar in (Bar("A", "20200101", 100), Bar("A", "2020-01-01", True)):
            with self.assertRaises(ValueError):
                Panel([bar])
        panel = synthetic_panel(1, 20)
        with self.assertRaises(ValueError):
            panel.label("S000", panel.dates[10], -1, panel.dates[15])
