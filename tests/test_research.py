import unittest
from datetime import date, datetime, timezone
from unittest.mock import Mock, patch

import pandas as pd

from engine.features import build_daily_features, build_outcome_labels
from engine.sources import fetch_reddit_history
from engine.text import extract_tickers, normalize_ticker


class TextTests(unittest.TestCase):
    def test_explicit_cashtags(self):
        self.assertEqual(extract_tickers("Watching $GME and $KOSS", "$CEO is noise"), ["GME", "KOSS"])

    def test_bare_mentions_are_limited_to_market_universe(self):
        self.assertEqual(
            extract_tickers("GME and AMC are moving", "CEO says KOSS next", {"GME", "KOSS"}),
            ["GME", "KOSS"],
        )

    def test_normalization(self):
        self.assertEqual(normalize_ticker("$amc"), "AMC")
        self.assertIsNone(normalize_ticker("CEO"))


class LabelTests(unittest.TestCase):
    @staticmethod
    def frame(highs, lows):
        return pd.DataFrame({"ticker":"TEST","session_date":pd.date_range("2020-01-01",periods=len(highs),freq="B"),"open":[10.0]*len(highs),"high":highs,"low":lows,"close":[10.0]*len(highs),"volume":[1000]*len(highs)})

    def test_clean_target_precedes_adverse(self):
        labels = build_outcome_labels(self.frame([10,16,10,10,10,10],[10,9,7,10,10,10]),horizon=5)
        self.assertEqual(labels.iloc[0]["outcome_class"], "clean_50")

    def test_same_bar_is_ambiguous(self):
        labels = build_outcome_labels(self.frame([10,16,10,10,10,10],[10,7,10,10,10,10]),horizon=5)
        self.assertEqual(labels.iloc[0]["outcome_class"], "ambiguous_50")


class FeatureTests(unittest.TestCase):
    def test_future_mentions_do_not_enter_prior_feature(self):
        bars = pd.DataFrame({"ticker":["GME"]*4,"session_date":pd.to_datetime(["2021-01-04","2021-01-05","2021-01-06","2021-01-07"]),"close":[10]*4,"volume":[100]*4})
        mentions = pd.DataFrame({"ticker":["GME"],"created_at":[datetime(2021,1,7,13,tzinfo=timezone.utc)],"author":["u1"],"community":["pennystocks"]})
        features = build_daily_features(bars, mentions)
        self.assertEqual(int(features.iloc[0]["mentions_3d"]), 0)
        self.assertEqual(int(features.iloc[-1]["mentions_1d"]), 1)

    def test_after_close_mention_is_excluded(self):
        bars = pd.DataFrame({"ticker":["GME"],"session_date":pd.to_datetime(["2021-01-07"]),"close":[10],"volume":[100]})
        mentions = pd.DataFrame({"ticker":["GME"],"created_at":[datetime(2021,1,8,1,tzinfo=timezone.utc)],"author":["u1"],"community":["pennystocks"]})
        features = build_daily_features(bars, mentions)
        self.assertEqual(int(features.iloc[0]["mentions_1d"]), 0)


class RedditSourceTests(unittest.TestCase):
    @patch("engine.sources.requests.get")
    def test_archive_window_uses_epoch_seconds(self, mock_get):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": []}
        mock_get.return_value = response

        posts, warnings = fetch_reddit_history(
            "pennystocks", date(2021, 1, 15), date(2021, 1, 15)
        )

        self.assertEqual(posts, [])
        self.assertEqual(warnings, [])
        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(params["after"], 1610668800)
        self.assertEqual(params["before"], 1610755200)


if __name__ == "__main__":
    unittest.main()
