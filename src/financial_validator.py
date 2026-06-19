from __future__ import annotations

import pandas as pd


def validate_financial_dataset(financial: pd.DataFrame) -> pd.DataFrame:
    if financial.empty:
        return pd.DataFrame(columns=["firm", "year", "rule", "status", "message"])
    wide = financial.pivot_table(index=["firm", "year"], columns="indicator_id", values="value", aggfunc="first").reset_index()
    rows: list[dict] = []
    for _, row in wide.iterrows():
        firm = row.get("firm")
        year = row.get("year")
        _check_gte(rows, firm, year, row, "TOTAL_ASSETS", "EQUITY", "Tổng tài sản >= Vốn chủ sở hữu")
        _check_gte(rows, firm, year, row, "TOTAL_ASSETS", "TOTAL_LIABILITIES", "Tổng tài sản >= Nợ phải trả")
        _check_nonnegative(rows, firm, year, row, "REVENUE", "Doanh thu >= 0")
        _check_nonnegative(rows, firm, year, row, "EMPLOYEES", "Số lao động >= 0")
    return pd.DataFrame(rows)


def attach_validation_flags(financial: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    output = financial.copy()
    output["validation_flag"] = ""
    if validation.empty:
        return output
    bad = validation.loc[validation["status"] == "FAIL"]
    for _, issue in bad.iterrows():
        mask = (output["firm"] == issue["firm"]) & (output["year"] == issue["year"])
        current = output.loc[mask, "validation_flag"].fillna("")
        flag = str(issue["rule"])
        output.loc[mask, "validation_flag"] = current.apply(lambda value: flag if not value else f"{value}; {flag}")
    return output


def _check_gte(rows: list[dict], firm, year, row: pd.Series, left: str, right: str, message: str) -> None:
    left_value = row.get(left)
    right_value = row.get(right)
    if pd.isna(left_value) or pd.isna(right_value):
        rows.append({"firm": firm, "year": year, "rule": f"{left}_GTE_{right}", "status": "SKIP", "message": "missing value"})
        return
    status = "PASS" if float(left_value) >= float(right_value) else "FAIL"
    rows.append({"firm": firm, "year": year, "rule": f"{left}_GTE_{right}", "status": status, "message": message})


def _check_nonnegative(rows: list[dict], firm, year, row: pd.Series, indicator: str, message: str) -> None:
    value = row.get(indicator)
    if pd.isna(value):
        rows.append({"firm": firm, "year": year, "rule": f"{indicator}_NONNEGATIVE", "status": "SKIP", "message": "missing value"})
        return
    status = "PASS" if float(value) >= 0 else "FAIL"
    rows.append({"firm": firm, "year": year, "rule": f"{indicator}_NONNEGATIVE", "status": status, "message": message})
