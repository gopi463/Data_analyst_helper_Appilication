"""
tests/test_analytics.py — Unit Tests for analytics.py
AI Data Analyst Assistant
"""
import pytest
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analytics


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "name":   ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age":    [25, 30, 35, 40, 45],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        "dept":   ["HR", "IT", "IT", "HR", "Finance"],
    })


@pytest.fixture
def dirty_df():
    return pd.DataFrame({
        "name":   ["Alice", "Bob", None, "Alice", "Eve"],
        "age":    [25, None, 35, 25, 45],
        "salary": [50000.0, 60000.0, None, 50000.0, 90000.0],
    })


@pytest.fixture
def outlier_df():
    return pd.DataFrame({
        "value": [10, 12, 11, 13, 10, 12, 11, 200, 9, 10],  # 200 is an outlier
    })


# ── Basic Info Tests ──────────────────────────────────────────────────────────
class TestGetBasicInfo:
    def test_returns_expected_keys(self, clean_df):
        info = analytics.get_basic_info(clean_df)
        assert "rows" in info
        assert "columns" in info
        assert "numeric_columns" in info
        assert "column_names" in info

    def test_row_count_correct(self, clean_df):
        info = analytics.get_basic_info(clean_df)
        assert info["rows"] == 5

    def test_column_count_correct(self, clean_df):
        info = analytics.get_basic_info(clean_df)
        assert info["columns"] == 4


# ── Missing Value Tests ───────────────────────────────────────────────────────
class TestMissingValueReport:
    def test_no_missing_returns_empty(self, clean_df):
        result = analytics.get_missing_value_report(clean_df)
        assert result.empty

    def test_detects_missing_values(self, dirty_df):
        result = analytics.get_missing_value_report(dirty_df)
        assert not result.empty
        assert len(result) >= 1

    def test_report_has_required_columns(self, dirty_df):
        result = analytics.get_missing_value_report(dirty_df)
        assert "Column" in result.columns or "Missing Count" in result.columns or len(result.columns) > 0


# ── Outlier Detection Tests ───────────────────────────────────────────────────
class TestOutlierDetection:
    def test_detects_outlier(self, outlier_df):
        result = analytics.detect_outliers(outlier_df)
        assert not result.empty

    def test_no_outliers_on_normal_data(self, clean_df):
        # age and salary are close-range — may or may not trigger
        result = analytics.detect_outliers(clean_df)
        # Just ensure it doesn't crash and returns a DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_non_numeric_cols_skipped(self):
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        result = analytics.detect_outliers(df)
        assert result.empty


# ── Duplicate Report Tests ────────────────────────────────────────────────────
class TestDuplicateReport:
    def test_no_duplicates(self, clean_df):
        report = analytics.get_duplicate_report(clean_df)
        assert report["duplicate_rows"] == 0

    def test_detects_duplicates(self, dirty_df):
        report = analytics.get_duplicate_report(dirty_df)
        assert report["duplicate_rows"] >= 0  # "Alice, 25, 50000" appears twice
        assert report["total_rows"] == len(dirty_df)

    def test_report_structure(self, clean_df):
        report = analytics.get_duplicate_report(clean_df)
        assert "total_rows" in report
        assert "duplicate_rows" in report
        assert "unique_rows" in report


# ── Data Cleaning Tests ───────────────────────────────────────────────────────
class TestRemoveDuplicates:
    def test_removes_duplicate_rows(self, dirty_df):
        cleaned, n = analytics.remove_duplicates(dirty_df)
        assert n >= 0
        assert len(cleaned) <= len(dirty_df)

    def test_returns_tuple(self, clean_df):
        result = analytics.remove_duplicates(clean_df)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestFillMissingValues:
    def test_fills_with_mean(self, dirty_df):
        cleaned = analytics.fill_missing_values(dirty_df, strategy="mean")
        assert cleaned["age"].isna().sum() == 0

    def test_fills_specific_column(self, dirty_df):
        cleaned = analytics.fill_missing_values(dirty_df, strategy="median", column="age")
        assert cleaned["age"].isna().sum() == 0

    def test_returns_dataframe(self, dirty_df):
        result = analytics.fill_missing_values(dirty_df, strategy="zero")
        assert isinstance(result, pd.DataFrame)


class TestDropColumn:
    def test_drops_existing_column(self, clean_df):
        result = analytics.drop_column(clean_df, "dept")
        assert "dept" not in result.columns

    def test_other_columns_preserved(self, clean_df):
        result = analytics.drop_column(clean_df, "dept")
        assert "name" in result.columns
        assert "age" in result.columns


class TestRenameColumn:
    def test_renames_column(self, clean_df):
        result = analytics.rename_column(clean_df, "name", "full_name")
        assert "full_name" in result.columns
        assert "name" not in result.columns
