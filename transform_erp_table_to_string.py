from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import re


@dataclass
class ReportTransformer:
    path: str
    sheet_name: Optional[str] = None

    # dd.mm.yyyy
    date_regex: re.Pattern = re.compile(r"^\s*\d{2}\.\d{2}\.\d{4}\s*$")

    def load(self) -> pd.DataFrame:
        """Загружает Excel 'как есть' (без заголовков)."""
        return pd.read_excel(
            self.path,
            sheet_name=self.sheet_name,
            header=None,
            engine="openpyxl",
        )

    def _find_start_row_by_date_in_first_col(self, raw: pd.DataFrame) -> int:
        """Находит первую строку данных по дате в колонке 0."""
        col0 = raw.iloc[:, 0].astype("string")
        mask = col0.str.match(self.date_regex, na=False)
        return int(mask.idxmax()) if mask.any() else 0

    @staticmethod
    def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
        """
        Нормализация строк:
        - \r\n/\r -> \n
        - strip
        - пустое -> NaN
        """
        df = df.copy()
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        if len(str_cols) == 0:
            return df

        def norm_series(col: pd.Series) -> pd.Series:
            s = col.astype("string")
            s = s.str.replace("\r\n", "\n", regex=False).str.replace("\r", "\n", regex=False)
            s = s.str.strip()
            s = s.mask(s.eq(""), other=pd.NA)
            return s

        df[str_cols] = df[str_cols].apply(norm_series)
        return df

    def _expand_multiline_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int]]:
        """
        Разворачивает колонки с переносами строк (\n) в несколько колонок.
        Параллельно считает max_lines (максимум строк в ячейке) для каждой исходной колонки.
        """
        df = df.copy()
        max_lines_by_col: Dict[int, int] = {}

        str_cols = df.select_dtypes(include=["object", "string"]).columns

        for c in list(str_cols):
            col = df[c].astype("string")

            if not col.str.contains("\n", na=False).any():
                continue

            line_counts = col.str.count("\n")
            max_lines = int((line_counts.dropna() + 1).max()) if line_counts.notna().any() else 1
            # c может быть int или str — для меты приведём к int если можно
            try:
                max_lines_by_col[int(c)] = max_lines
            except Exception:
                max_lines_by_col[str(c)] = max_lines

            split_df = col.str.split("\n", expand=True).astype("string")
            split_df = split_df.apply(lambda x: x.str.strip())
            split_df = split_df.mask(split_df.eq(""), other=pd.NA)

            split_df.columns = [f"{c}__{i+1}" for i in range(split_df.shape[1])]

            insert_at = df.columns.get_loc(c)
            df = df.drop(columns=[c])
            left = df.iloc[:, :insert_at]
            right = df.iloc[:, insert_at:]
            df = pd.concat([left, split_df, right], axis=1)

        df = df.dropna(axis=1, how="all")
        return df, max_lines_by_col

    def _detect_date_column(self, df: pd.DataFrame):
        """Определяет колонку даты по доле значений, подходящих под dd.mm.yyyy."""
        best_col = None
        best_ratio = -1.0

        for c in df.columns:
            col = df[c].astype("string")
            ratio = float(col.str.match(self.date_regex, na=False).mean())
            if ratio > best_ratio:
                best_ratio = ratio
                best_col = c

        return best_col

    def transform_to_csv(self, export_csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Делает трансформацию и сохраняет CSV.
        Возвращает df (в памяти) и meta.
        """
        raw = self.load()

        start_idx = self._find_start_row_by_date_in_first_col(raw)
        df = raw.iloc[start_idx:].copy()

        # чистим пустое
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)

        # нормализация
        df = self._normalize_strings(df)

        # разворот многострочных
        df, max_lines_by_col = self._expand_multiline_columns(df)

        # повторная чистка
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

        # дата
        date_col = self._detect_date_column(df)
        if date_col is None:
            raise ValueError("Не удалось определить колонку с датой (dd.mm.yyyy).")

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", format="%d.%m.%Y")
        df = df.dropna(subset=[date_col]).reset_index(drop=True)

        # переименования
        other_cols = [c for c in df.columns if c != date_col]
        rename_map = {date_col: "дата"}
        rename_map.update({c: str(i) for i, c in enumerate(other_cols, start=1)})
        df = df.rename(columns=rename_map)

        # финальная чистка
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)

        # CSV
        df.to_csv(export_csv_path, index=False, encoding="utf-8-sig")

        meta = {
            "source_path": self.path,
            "sheet_name": self.sheet_name,
            "start_row_index_in_source": start_idx,
            "max_lines_by_original_col": max_lines_by_col,
            "date_col_detected": str(date_col),
            "shape": df.shape,
            "csv_path": export_csv_path,
        }
        return df, meta


def main():
    # ВХОД: Excel из 1С
    xlsx_path = r"Z:\01 Администрация\Кобычев\МК P&L\[MK P&L] Отчет по проводкам.xlsx"

    # ВЫХОД: CSV (на него потом будет ссылаться другой скрипт)
    csv_path = r"Z:\01 Администрация\Кобычев\МК P&L\report_transformed.csv"

    transformer = ReportTransformer(path=xlsx_path, sheet_name="Лист_1")

    df, meta = transformer.transform_to_csv(export_csv_path=csv_path)

    # Показать в консоли, что всё ок
    print("\n=== CSV сформирован ===")
    print("Путь:", meta["csv_path"])
    print("Размер (строк, колонок):", meta["shape"])
    print("Колонки:", list(df.columns))

    print("\n=== Первые 10 строк (preview) ===")
    # Чтобы не резало по ширине
    with pd.option_context("display.max_columns", 30, "display.width", 160):
        print(df.head(10))

    print("\n=== META ===")
    print(meta)


if __name__ == "__main__":
    main()
