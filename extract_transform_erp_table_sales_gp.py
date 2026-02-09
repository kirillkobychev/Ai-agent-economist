# extract_transform_erp_table_sales_gp.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, List

import numpy as np
import pandas as pd


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class SalesGPCorrectionConfig:
    input_filename: str = "[MK P&L] Выручка и себестоимость ГП и товары.xlsx"
    additional_filenames: Tuple[str, ...] = (
        "[MK P&L] Выручка и себестоимость 45",
        "[MK P&L] Выручка услуги",
    )

    base_dir: Optional[Path] = None
    sheet_name: Optional[str] = None

    # Колонки
    doc_col: str = "Товары.Ссылка"
    counterparty_col: str = "Товары.Ссылка.Контрагент"
    nomenclature_col: str = "Товары.Номенклатура"
    volume_col: str = "Объем отгрузки, тн"
    cost_col: str = "Сумма СС"

    # Разметка
    direction_col: str = "Документ.Направление деятельности"
    sku_group_col: str = "SKU GROUP NAME"

    # для “устойчивого” режима 43*
    gfu_col: str = "ГФУ"
    analytic_group_col: str = "Группа аналит.учета"
    gfu_finished_value: str = "43* Готовая продукция"

    # для услуг
    fin_group_col_main: str = "Документ.Заказ клиента.Группа фин. учета расчетов"
    fin_group_col_services: str = "Товары.Заказ клиента.Группа фин. учета расчетов"

    # Fix себестоимости: только реализации
    doc_type_substring: str = "Реализация товаров и услуг"

    # Ключ для поиска задвоенной/затроенной общей СС
    key_cols: Tuple[str, ...] = ("Товары.Ссылка", "Товары.Ссылка.Контрагент", "Товары.Номенклатура")

    eps: float = 1e-12
    output_suffix: str = "__fixed_sum_ss"


# ----------------------------
# Utils
# ----------------------------
class NumericParser:
    @staticmethod
    def to_float(series: pd.Series) -> pd.Series:
        s = series.astype("string")
        s = s.str.replace("\u00A0", "", regex=False)  # NBSP
        s = s.str.replace(" ", "", regex=False)
        s = s.str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce").fillna(0.0)


class ExcelPathResolver:
    def __init__(self, cfg: SalesGPCorrectionConfig):
        self.cfg = cfg

    @staticmethod
    def _ensure_xlsx_name(name: str) -> List[str]:
        name = name.strip()
        if name.lower().endswith((".xlsx", ".xls")):
            return [name]
        return [f"{name}.xlsx", name]

    def resolve_main_input_path(self) -> Path:
        fname = self.cfg.input_filename
        candidates: List[Path] = []

        if self.cfg.base_dir is not None:
            candidates.append(self.cfg.base_dir / fname)

        candidates.append(Path(__file__).resolve().parent / fname)
        candidates.append(Path(r"Z:\01 Администрация\Кобычев\МК P&L") / fname)

        for p in candidates:
            if p.exists():
                return p

        tried = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Не найден входной Excel. Пробовал пути:\n{tried}")

    def resolve_additional_paths(self, base_dir: Path) -> List[Path]:
        paths: List[Path] = []
        for raw in self.cfg.additional_filenames:
            found = None
            for nm in self._ensure_xlsx_name(raw):
                cand = base_dir / nm
                if cand.exists():
                    found = cand
                    break
            if found is None:
                tried = "\n".join(str(base_dir / nm) for nm in self._ensure_xlsx_name(raw))
                raise FileNotFoundError(f"Не найден дополнительный Excel: '{raw}'. Пробовал:\n{tried}")
            paths.append(found)
        return paths

    @staticmethod
    def make_output_path(input_path: Path, suffix: str) -> Path:
        return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


class ExcelIO:
    @staticmethod
    def detect_first_sheet(path: Path) -> str:
        from openpyxl import load_workbook

        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            return wb.sheetnames[0]
        finally:
            wb.close()

    @staticmethod
    def read_excel(path: Path, sheet_name: Optional[str]) -> tuple[pd.DataFrame, str]:
        use_sheet = sheet_name or ExcelIO.detect_first_sheet(path)
        df = pd.read_excel(path, sheet_name=use_sheet, engine="openpyxl")
        return df, use_sheet

    @staticmethod
    def write_excel(path: Path, df: pd.DataFrame, sheet_name: str) -> None:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        try:
            from openpyxl import load_workbook

            wb = load_workbook(path)
            ws = wb[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            wb.save(path)
        except Exception:
            pass


class TableUnionByBaseColumns:
    @staticmethod
    def union(base_df: pd.DataFrame, other_dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
        base_cols = list(base_df.columns)
        aligned: List[pd.DataFrame] = [base_df.copy()]
        for odf in other_dfs:
            aligned.append(odf.reindex(columns=base_cols))
        return pd.concat(aligned, axis=0, ignore_index=True)


# ----------------------------
# Pipeline base
# ----------------------------
class PipelineStep:
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class PandasPipeline:
    def __init__(self, steps: Sequence[PipelineStep]):
        self.steps = list(steps)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            df = step.run(df)
        return df


# ----------------------------
# Step 1: Fix duplicated cost in реализация
# ----------------------------
class RedistributeDuplicatedCostByVolumeStep(PipelineStep):
    def __init__(
        self,
        *,
        doc_col: str,
        doc_type_substring: str,
        key_cols: Sequence[str],
        volume_col: str,
        cost_col: str,
        eps: float = 1e-12,
    ):
        self.doc_col = doc_col
        self.doc_type_substring = doc_type_substring
        self.key_cols = list(key_cols)
        self.volume_col = volume_col
        self.cost_col = cost_col
        self.eps = float(eps)

        self._vol_tmp = "__vol_num"
        self._cost_tmp = "__cost_num"
        self._cnz_tmp = "__cost_nz_flag"
        self._cnzv_tmp = "__cost_nz_val"

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        need_cols = set(self.key_cols + [self.doc_col, self.volume_col, self.cost_col])
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise KeyError(f"В Excel не найдены обязательные колонки: {missing}")

        out = df.copy()

        doc_mask = (
            out[self.doc_col]
            .astype("string")
            .str.contains(self.doc_type_substring, case=False, na=False)
        )

        vol = NumericParser.to_float(out[self.volume_col])
        cost = NumericParser.to_float(out[self.cost_col])

        out[self._vol_tmp] = vol
        out[self._cost_tmp] = cost

        nonzero = cost.abs() > self.eps
        out[self._cnz_tmp] = nonzero.astype("int8")
        out[self._cnzv_tmp] = cost.where(nonzero, np.nan)

        g = out.groupby(self.key_cols, dropna=False, sort=False)

        group_size = g[self._cost_tmp].transform("size").astype("int64")
        cost_nonzero_count = g[self._cnz_tmp].transform("sum").astype("int64")
        cost_nz_min = g[self._cnzv_tmp].transform("min")
        cost_nz_max = g[self._cnzv_tmp].transform("max")

        fix_mask = (
            doc_mask
            & (group_size > 1)
            & (cost_nonzero_count == group_size)
            & (cost_nz_min == cost_nz_max)
        )

        if not bool(fix_mask.any()):
            out[self.cost_col] = cost
            out.drop(columns=[self._vol_tmp, self._cost_tmp, self._cnz_tmp, self._cnzv_tmp], inplace=True, errors="ignore")
            return out

        total_cost = cost_nz_max.fillna(0.0)
        total_vol = g[self._vol_tmp].transform("sum")

        corrected = cost.to_numpy(copy=True)
        fix_np = fix_mask.to_numpy()

        total_vol_np = total_vol.to_numpy()
        vol_np = vol.to_numpy()
        total_cost_np = total_cost.to_numpy()
        group_size_np = group_size.to_numpy()

        vol_ok = np.abs(total_vol_np) > self.eps

        idx1 = fix_np & vol_ok
        corrected[idx1] = total_cost_np[idx1] * vol_np[idx1] / total_vol_np[idx1]

        idx2 = fix_np & (~vol_ok)
        corrected[idx2] = total_cost_np[idx2] / group_size_np[idx2]

        out[self.cost_col] = corrected
        out.drop(columns=[self._vol_tmp, self._cost_tmp, self._cnz_tmp, self._cnzv_tmp], inplace=True, errors="ignore")
        return out


# ----------------------------
# Step 2: SKU GROUP NAME = winner by frequency (plant)
# ----------------------------
class AddSkuGroupNameStep(PipelineStep):
    """
    Делает ровно то, что ты попросил:

    Для каждой номенклатуры считаем частоты по заводам (Фибратек/ЛАТО/Техпром)
    на “надежных” метках (direction=завод ИЛИ сильные заводские overrides).

    Для заводских SKU (префиксы ДН/ДТ/ЛПН/ЛПП/ДФТ/… ИЛИ 43* + нужные группы аналитики)
    ставим winner_plant (самый частый завод по этой номенклатуре).

    Закупные/сторонние (люки, МПК, асбокартон, грабли, RIMROOF и т.п.) жёстко -> Прочие.
    Услуги (“Покупатели ...”) залочены.
    """

    def __init__(
        self,
        *,
        direction_col: str,
        nomenclature_col: str,
        output_col: str,
        gfu_col: str,
        analytic_group_col: str,
        gfu_finished_value: str,
        fin_group_col_main: str,
    ):
        self.direction_col = direction_col
        self.nomenclature_col = nomenclature_col
        self.output_col = output_col
        self.gfu_col = gfu_col
        self.analytic_group_col = analytic_group_col
        self.gfu_finished_value = gfu_finished_value
        self.fin_group_col_main = fin_group_col_main

        self._plant_labels: List[str] = ["Фибратек", "ЛАТО", "Техпром"]

        # seeds по направлению (ТОЛЬКО заводы)
        self._dir_map_seeds: Dict[str, str] = {
            "Площадка Фибратек МК": "Фибратек",
            "Площадка Лато МК": "ЛАТО",
            "Площадка Техпром МК": "Техпром",
        }

        # --- заводские overrides (сильные маркеры) ---
        self._plant_override_rules: List[tuple[str, str]] = [
            (r"\bфибратек\b", "Фибратек"),
            (r"\bfibra\s*plank\b", "ЛАТО"),
            (r"\(.*лато.*\)", "ЛАТО"),
            (r"\(.*фибратек.*\)", "Фибратек"),
            (r"\(.*техпром.*\)", "Техпром"),
        ]

        # --- закупные overrides -> Прочие ---
        self._prochie_rules: List[str] = [
            r"\brimroof\b",
            r"\b(?:асбокартон|каон)\b",
            r"\bграбл[а-я]*\b",
            r"\bуголок\b.*\bмпк\b",
            r"\bлюк\b.*\bчугун\b",
            r"\bлюк\s+полимерно-песчаный\b",
            r"\bплита\s+чугунная\b.*\bбежецклмз\b",
            r"\bдоска\s+террасная\b.*\bмпк\b",
            r"\bfadoco\b",
            r"\bfachmann\b",
            r"\bgarten\b",
            r"\bexperte\b",
            r"\b(?:опрыскивател|распылител|пистолет-распылител|коннектор|аквастоп|мотыга|шампур)\b",
            r"\bпрофиль\b.*\bалюмини\b",
        ]

        # заводские префиксы (добавил ДТ)
        self._manufacturing_pat = (
            r"^\s*(?:ЛПН|ЛПП|ДН|ДТ|БТ|БНТ|ВТ|ДФТ)\b|"
            r"^\s*Волна-|"
            r"\bFIBRA\s*PLANK\b"
        )

        self._analytic_dir_priority_set = {
            "лпп ту",
            "лпн",
            "дн",
            "лпп",
            "лпн 16-40 мм",
            "сайдинг текстура",
            "ппфго",
            "сайдинг текстура грунт.",
            "текстура",
            "сайдинг текстура окраш.",
            "sidwood",
            "сайдинг текстура грунт. бц",
            "сайдинг текстура окраш. в массе грунт.",
            "сайдинг текстура окраш. в массе",
        }

    @staticmethod
    def _norm_str(s: pd.Series) -> pd.Series:
        return (
            s.astype("string")
            .str.replace("\u00A0", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    @staticmethod
    def _clean_existing_label(s: pd.Series) -> pd.Series:
        x = s.astype("string").str.strip()
        return x.mask(x.eq(""), pd.NA)

    @staticmethod
    def _casefold_series(s: pd.Series) -> pd.Series:
        return s.astype("string").str.casefold()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in [self.direction_col, self.nomenclature_col] if c not in df.columns]
        if missing:
            raise KeyError(f"В Excel не найдены обязательные колонки для разметки: {missing}")

        out = df.copy()

        # existing SKU (если уже есть)
        if self.output_col in out.columns:
            existing = self._clean_existing_label(out[self.output_col])
        else:
            existing = pd.Series(pd.NA, index=out.index, dtype="string")

        # lock только для услуг ("Покупатели ...")
        locked = pd.Series(False, index=out.index)
        if self.fin_group_col_main in out.columns:
            fin_group = self._norm_str(out[self.fin_group_col_main])
            is_services_row = fin_group.str.startswith("Покупатели", na=False)
            locked = existing.notna() & is_services_row
        can_edit = ~locked

        direction = self._norm_str(out[self.direction_col])
        nomen = self._norm_str(out[self.nomenclature_col].astype("string"))
        n = nomen.fillna("")
        n_cf = self._casefold_series(n)

        # маски
        is_manufacturing = n.str.contains(self._manufacturing_pat, case=False, na=False, regex=True)

        stable_43_mask = pd.Series(False, index=out.index)
        if self.gfu_col in out.columns and self.analytic_group_col in out.columns:
            gfu = self._norm_str(out[self.gfu_col])
            ag = self._norm_str(out[self.analytic_group_col])
            ag_cf = self._casefold_series(ag).str.replace(r"\s+", " ", regex=True).str.strip()
            is_gfu_43 = gfu.eq(self.gfu_finished_value)
            ag_match = ag_cf.isin(self._analytic_dir_priority_set) | ag_cf.str.startswith("волна", na=False)
            stable_43_mask = is_gfu_43 & ag_match

        # ---------
        # 1) PROCHIE overrides mask
        # ---------
        prochie_mask = pd.Series(False, index=out.index)
        for pat in self._prochie_rules:
            prochie_mask |= n_cf.str.contains(pat, na=False, regex=True)

        # ---------
        # 2) Plant seeds from direction + plant overrides
        # ---------
        seed_plant = direction.map(self._dir_map_seeds)

        # fallback по подстроке (на случай вариаций текста)
        need = seed_plant.isna()
        if bool(need.any()):
            d = direction.fillna("")
            seed_plant = seed_plant.mask(need & d.str.contains("Фибратек", case=False, na=False), "Фибратек")
            seed_plant = seed_plant.mask(need & d.str.contains("Лато", case=False, na=False), "ЛАТО")
            seed_plant = seed_plant.mask(need & d.str.contains("Техпром", case=False, na=False), "Техпром")

        # plant_override из номенклатуры
        plant_override = pd.Series(pd.NA, index=out.index, dtype="string")
        for pat, lbl in self._plant_override_rules:
            m = n_cf.str.contains(pat.casefold(), na=False, regex=True)
            plant_override = plant_override.mask(m & plant_override.isna(), lbl)

        # надежная метка завода для обучения частот:
        # - только завод (seed_plant / plant_override)
        # - и НЕ закупное (не prochie_mask)
        plant_train = seed_plant.copy()
        plant_train = plant_train.mask(plant_override.notna(), plant_override)
        plant_train = plant_train.mask(prochie_mask, pd.NA)  # закупное не участвует

        # ---------
        # 3) Frequency map: nomenclature -> winner plant
        # ---------
        train_mask = plant_train.isin(self._plant_labels) & nomen.notna()
        if bool(train_mask.any()):
            tmp = pd.DataFrame(
                {
                    "_nom": nomen.loc[train_mask].to_numpy(),
                    "_plant": plant_train.loc[train_mask].astype("string").to_numpy(),
                }
            )
            freq = (
                tmp.groupby(["_nom", "_plant"], dropna=False, sort=False)
                .size()
                .unstack(fill_value=0)
            )
            # ensure all plant columns exist
            for lbl in self._plant_labels:
                if lbl not in freq.columns:
                    freq[lbl] = 0
            freq = freq[self._plant_labels]

            nom_to_winner_plant: Dict[str, str] = freq.idxmax(axis=1).to_dict()
        else:
            nom_to_winner_plant = {}

        # plant_prior на случай полной новой номенклатуры без истории
        plant_prior = "Фибратек"
        vc = plant_train[plant_train.isin(self._plant_labels)].value_counts()
        if not vc.empty:
            plant_prior = str(vc.idxmax())

        winner_plant = nomen.map(nom_to_winner_plant)

        # ---------
        # 4) Final assignment
        # ---------
        final = existing.copy()

        # (A) закупные всегда Прочие (если можно редактировать)
        final = final.mask(can_edit & prochie_mask, "Прочие")

        # (B) сильный override завода
        final = final.mask(can_edit & final.isna() & plant_override.notna(), plant_override)

        # (C) заводские строки: ставим победителя по частоте
        #     (включая случаи, когда направление пусто/прочие)
        plant_needed = can_edit & (is_manufacturing | stable_43_mask) & (~prochie_mask)
        final = final.mask(plant_needed & winner_plant.notna(), winner_plant)
        final = final.mask(plant_needed & winner_plant.isna() & seed_plant.notna(), seed_plant)
        final = final.mask(plant_needed & winner_plant.isna() & seed_plant.isna(), plant_prior)

        # (D) остальные: если нет — fallback
        final = final.mask(can_edit & final.isna() & seed_plant.notna(), seed_plant)
        final = final.fillna("Прочие")

        out[self.output_col] = final
        cols = [c for c in out.columns if c != self.output_col] + [self.output_col]
        return out[cols]


# ----------------------------
# Services preprocessing
# ----------------------------
class ServicesPreprocessor:
    def __init__(self, *, fin_group_col_main: str, fin_group_col_services: str, sku_group_col: str):
        self.fin_group_col_main = fin_group_col_main
        self.fin_group_col_services = fin_group_col_services
        self.sku_group_col = sku_group_col

        self._map: Dict[str, str] = {
            "Покупатели (Продукция Фибратек)": "Фибратек",
            "Покупатели (Продукция ЛАТО)": "ЛАТО",
            "Покупатели (Продукция МК)": "Прочие",
            "Покупатели (Продукция СПБ)": "Прочие",
            "Покупатели (Продукция ТЕХПРОМ)": "Техпром",
            "Покупатели (Аренда)": "Прочие",
            "Покупатели (группа)": "Прочие",
        }

    @staticmethod
    def _norm_series(s: pd.Series) -> pd.Series:
        return (
            s.astype("string")
            .str.replace("\u00A0", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if self.fin_group_col_services in out.columns and self.fin_group_col_main not in out.columns:
            out = out.rename(columns={self.fin_group_col_services: self.fin_group_col_main})
        elif self.fin_group_col_services in out.columns and self.fin_group_col_main in out.columns:
            main = out[self.fin_group_col_main]
            srv = out[self.fin_group_col_services]
            out[self.fin_group_col_main] = main.where(main.notna(), srv)
            out = out.drop(columns=[self.fin_group_col_services])

        if self.sku_group_col not in out.columns:
            out[self.sku_group_col] = pd.NA

        if self.fin_group_col_main in out.columns:
            g = self._norm_series(out[self.fin_group_col_main])
            out[self.sku_group_col] = g.map(self._map).fillna("Прочие")
        else:
            out[self.sku_group_col] = "Прочие"

        return out


# ----------------------------
# Entrypoint
# ----------------------------
def main():
    cfg = SalesGPCorrectionConfig()
    resolver = ExcelPathResolver(cfg)

    main_path = resolver.resolve_main_input_path()
    base_dir = main_path.parent
    add_paths = resolver.resolve_additional_paths(base_dir)

    df_main, sheet_main = ExcelIO.read_excel(main_path, cfg.sheet_name)
    if cfg.sku_group_col not in df_main.columns:
        df_main[cfg.sku_group_col] = pd.NA

    svc_prep = ServicesPreprocessor(
        fin_group_col_main=cfg.fin_group_col_main,
        fin_group_col_services=cfg.fin_group_col_services,
        sku_group_col=cfg.sku_group_col,
    )

    other_dfs: List[pd.DataFrame] = []
    for p in add_paths:
        df_x, _ = ExcelIO.read_excel(p, cfg.sheet_name)

        if "услуг" in p.stem.lower():
            df_x = svc_prep.apply(df_x)

        if cfg.fin_group_col_services in df_x.columns and cfg.fin_group_col_main not in df_x.columns:
            df_x = df_x.rename(columns={cfg.fin_group_col_services: cfg.fin_group_col_main})

        if cfg.sku_group_col not in df_x.columns:
            df_x[cfg.sku_group_col] = pd.NA

        other_dfs.append(df_x)

    df_all = TableUnionByBaseColumns.union(df_main, other_dfs)

    pipeline = PandasPipeline(
        steps=[
            RedistributeDuplicatedCostByVolumeStep(
                doc_col=cfg.doc_col,
                doc_type_substring=cfg.doc_type_substring,
                key_cols=cfg.key_cols,
                volume_col=cfg.volume_col,
                cost_col=cfg.cost_col,
                eps=cfg.eps,
            ),
            AddSkuGroupNameStep(
                direction_col=cfg.direction_col,
                nomenclature_col=cfg.nomenclature_col,
                output_col=cfg.sku_group_col,
                gfu_col=cfg.gfu_col,
                analytic_group_col=cfg.analytic_group_col,
                gfu_finished_value=cfg.gfu_finished_value,
                fin_group_col_main=cfg.fin_group_col_main,
            ),
        ]
    )

    out_df = pipeline.run(df_all)

    output_path = resolver.make_output_path(main_path, cfg.output_suffix)
    ExcelIO.write_excel(output_path, out_df, sheet_main)

    print("\n=== ГОТОВО: SKU GROUP NAME = завод по частоте встречаемости номенклатуры (для заводских SKU) ===")
    print("Output :", str(output_path))
    print("Rows   :", len(out_df))
    print("Cols   :", len(out_df.columns))


if __name__ == "__main__":
    main()
