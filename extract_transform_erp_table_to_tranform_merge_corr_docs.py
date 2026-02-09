from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import re


@dataclass
class CorrDocsByNomenclatureToExcel:
    input_csv_path: str
    output_excel_path: str

    # Колонки проводок
    debit_col: str = "13"
    credit_col: str = "15"
    amount_col: str = "16"

    # Дата из первого скрипта
    date_col: str = "дата"

    # Типы документов
    doc_types: Tuple[str, ...] = (
        "Возврат товаров от клиента",
        "Корректировка приобретения",
        "Корректировка реализации до ввода остатков",
        "Корректировка реализации",
        "Распределение расходов",
    )

    # Корреспонденции (префиксами)
    revenue_debit_prefix: str = "62.01"
    revenue_credit_prefix: str = "90.01.1"

    # НДС: 90.03 -> 68.02 / 19.*
    vat_debit_prefix: str = "90.03"
    vat_credit_prefixes: Tuple[str, ...] = ("68.02", "19.")

    # Себестоимость: 90.02.1 -> 41.* / 45.*
    cogs_debit_prefix: str = "90.02.1"
    cogs_credit_prefixes: Tuple[str, ...] = ("41.", "45.")

    # Признаки аналитик
    inv_group_re: re.Pattern = re.compile(r"^(43\*|41\*)\s*.*", flags=re.IGNORECASE)
    warehouse_re: re.Pattern = re.compile(r"^Склад\b", flags=re.IGNORECASE)
    platform_re: re.Pattern = re.compile(r"^Площадка\b", flags=re.IGNORECASE)

    # Контрагент
    counterparty_mark_re: re.Pattern = re.compile(r"\b(ООО|АО|ПАО|ИП|ЗАО|ОАО)\b", flags=re.IGNORECASE)
    private_person_re: re.Pattern = re.compile(r"^Частное лицо$", flags=re.IGNORECASE)
    fio_re: re.Pattern = re.compile(r"^[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){2}$")

    # CAPS-последовательности (ИНТЕРНЕТ РЕШЕНИЯ ГП / СТЕНА и т.п.)
    caps_seq_re: re.Pattern = re.compile(r"(?:[A-ZА-ЯЁ]{2,}(?:\s+[A-ZА-ЯЁ]{2,}){0,7})")

    # ---- Номенклатура: универсальные признаки ----
    dim_re: re.Pattern = re.compile(r"\d+\s*(?:[xх\*/]\s*\d+){1,}", flags=re.IGNORECASE)  # 1750*1130*5.2, 3000х1500х10
    units_re: re.Pattern = re.compile(r"\b(кг|г|мм|см|м|л|ведро|нагрузка|ral|тн)\b", flags=re.IGNORECASE)
    has_digit_re: re.Pattern = re.compile(r"\d")
    acct_like_re: re.Pattern = re.compile(r"^\d{2}(\.\d{2}(\.\d+)?)?$")

    # "товарные" слова (помогает для "Минеральная черепица RIMROOF серая" и т.п.)
    product_words_re: re.Pattern = re.compile(
        r"\b("
        r"черепица|краска|праймер|мастика|люк|лопата|грабли|дверка|духовка|лист|"
        r"волна|планк|профиль|панель|плита|плитка|шпатель|кисть|саморез|гвоздь|"
        r"пена|герметик|клей|шайба|болт|гайка|рулон|пленка|мембрана|уголок|"
        r"решетка|кран|смеситель|фитинг|труба|бита|сверло|диск"
        r")\b",
        flags=re.IGNORECASE,
    )

    # латиница/бренд
    latin_re: re.Pattern = re.compile(r"[A-Za-z]{3,}")

    # документ: номер и "от dd.mm.yyyy"
    doc_num_re: re.Pattern = re.compile(r"\b(МК\d{2}-\d{6}|\d{8,})\b", flags=re.IGNORECASE)
    doc_ot_date_re: re.Pattern = re.compile(r"\bот\s+\d{2}\.\d{2}\.\d{4}\b", flags=re.IGNORECASE)

    # числовой допуск для "нулей"
    eps: float = 1e-9

    # ---------- базовые утилиты ----------

    @staticmethod
    def _to_amount(series: pd.Series) -> pd.Series:
        s = series.astype("string").str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    @staticmethod
    def _split_tokens(value: Optional[str]) -> List[str]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        s = str(value).replace("\r\n", "\n").replace("\r", "\n")
        parts = [p.strip() for p in s.split("\n")]
        return [p for p in parts if p]

    @staticmethod
    def _pick_first_non_empty(series: pd.Series):
        s = series.astype("string").str.strip().replace({"": pd.NA}).dropna()
        return s.iloc[0] if len(s) else pd.NA

    @staticmethod
    def _uppercase_ratio(s: str) -> float:
        letters = re.findall(r"[A-Za-zА-Яа-яЁё]", s)
        if not letters:
            return 0.0
        upper = sum(1 for ch in letters if ch.isupper())
        return upper / max(len(letters), 1)

    @staticmethod
    def _normalize_nomenclature(tok: str) -> str:
        t = tok.strip()
        t = re.sub(r"^\d{1,2}\s*%\s*", "", t)  # убрать "20% "
        t = re.sub(r"\s{2,}", " ", t)
        return t.strip()

    def _startswith_any(self, s: pd.Series, prefixes: Tuple[str, ...]) -> pd.Series:
        out = pd.Series(False, index=s.index)
        for p in prefixes:
            out |= s.str.startswith(p, na=False)
        return out

    @staticmethod
    def _is_noise_token(tok: str) -> bool:
        t = str(tok).strip()
        if not t:
            return True
        if t in {"<...>", "<…>"}:
            return True
        if t.startswith("<") and t.endswith(">") and len(t) <= 10:
            return True
        return False

    def _contains_doc_type(self, s: str) -> bool:
        u = str(s).upper()
        return any(dt.upper() in u for dt in self.doc_types)

    def _looks_like_doc_header(self, s: str) -> bool:
        t = str(s)
        return bool(self._contains_doc_type(t) and self.doc_num_re.search(t) and self.doc_ot_date_re.search(t))

    # ---------- поиск колонки документа ----------

    def _detect_doc_text_col(self, df: pd.DataFrame) -> str:
        doc_list = sorted(self.doc_types, key=len, reverse=True)
        pattern = "|".join(re.escape(x) for x in doc_list)

        best_col = None
        best_ratio = 0.0
        for c in df.columns:
            col = df[c].astype("string").str.strip()
            ratio = float(col.str.contains(pattern, na=False).mean())
            if ratio > best_ratio:
                best_ratio = ratio
                best_col = c

        if not best_col or best_ratio == 0.0:
            if "1" in df.columns:
                return "1"
            raise ValueError("Не удалось найти колонку с названием документа (по doc_types).")
        return str(best_col)

    def _doc_header_re(self) -> re.Pattern:
        doc_list = sorted(self.doc_types, key=len, reverse=True)
        dt_alt = "|".join(re.escape(x) for x in doc_list)
        num_alt = r"(?:МК\d{2}-\d{6}|\d{8,})"
        return re.compile(
            rf"(?P<dt>{dt_alt})\s+(?P<num>{num_alt})\s+от\s+(?P<d>\d{{2}}\.\d{{2}}\.\d{{4}})\s+(?P<t>\d{{1,2}}:\d{{2}}:\d{{2}})",
            flags=re.IGNORECASE,
        )

    # ---------- scoring ----------

    def _score_nomenclature(self, token: str) -> int:
        t = str(token).strip()
        if not t or self._is_noise_token(t):
            return -999

        # запрет: заголовки документов / строки с типами документов
        if self._looks_like_doc_header(t) or self._contains_doc_type(t):
            return -500

        u = t.upper()
        score = 0

        # сильные признаки
        if self.dim_re.search(t):
            score += 12
        if self.units_re.search(t):
            score += 10
        if self.has_digit_re.search(t) and re.search(r"[A-Za-zА-Яа-яЁё]", t):
            score += 6

        # товарные слова (например "черепица ...")
        if self.product_words_re.search(t):
            score += 10

        # латиница/бренд (RIMROOF, FIBRA ...)
        if self.latin_re.search(t):
            score += 8

        # формат "описание товара"
        if any(ch in t for ch in ["(", ")", ",", ";", '"', "'"]):
            score += 2
        if len(t) >= 16:
            score += 2
        if len(t) >= 28:
            score += 2

        # штрафы
        if u.startswith(("СКЛАД", "ПЛОЩАДКА")):
            score -= 50
        if self.inv_group_re.match(t):
            score -= 60
        if self.counterparty_mark_re.search(t) or self.private_person_re.match(t) or self.fio_re.match(t):
            score -= 25
        if self.acct_like_re.match(t):
            score -= 60

        if any(w in u for w in ["ОСНОВНОЕ ПОДРАЗДЕЛЕНИЕ", "КОМИССИОННЫЙ", "ДОГОВОР"]):
            score -= 20

        return score

    def _score_counterparty(self, token: str) -> int:
        t = str(token).strip()
        if not t or self._is_noise_token(t):
            return -999
        if self._looks_like_doc_header(t):
            return -300

        u = t.upper()
        score = 0

        if self.private_person_re.match(t):
            return 30
        if self.fio_re.match(t):
            return 26

        has_legal_form = bool(self.counterparty_mark_re.search(t))
        if has_legal_form:
            score += 14

        if not re.search(r"\d", t) and len(t) >= 3:
            ur = self._uppercase_ratio(t)
            if " " not in t and ur >= 0.90 and len(t) >= 4:
                score += 16
            elif " " in t:
                if ur >= 0.85:
                    score += 16
                elif ur >= 0.70:
                    score += 10

        # бонус смешанного регистра + юрформа (ТехноНИКОЛЬ ... ООО)
        has_lower = bool(re.search(r"[a-zа-яё]", t))
        has_upper = bool(re.search(r"[A-ZА-ЯЁ]", t))
        if has_legal_form and has_lower and has_upper:
            score += 8

        if len(t) >= 10:
            score += 2
        if len(t) >= 20:
            score += 2

        if u.startswith(("СКЛАД", "ПЛОЩАДКА")):
            score -= 12
        if self.inv_group_re.match(t):
            score -= 12
        if self.dim_re.search(t):
            score -= 6
        if self.acct_like_re.match(t):
            score -= 10

        return score

    def _counterparty_subtokens(self, tok: str) -> List[str]:
        t = str(tok).strip()
        if not t or self._is_noise_token(t) or self._looks_like_doc_header(t):
            return []

        t_clean = re.sub(r'[\"“”<>]+', " ", t)
        t_clean = re.sub(r"\s{2,}", " ", t_clean).strip()

        out: List[str] = []

        if self.private_person_re.search(t_clean):
            out.append("Частное лицо")

        fio_inside = re.findall(r"[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){2}", t_clean)
        out.extend(fio_inside)

        bad = {"ПЛОЩАДКА", "ОСНОВНОЕ", "ПОДРАЗДЕЛЕНИЕ", "СКЛАД", "ДОГОВОР", "КОМИССИОННЫЙ"}
        for m in self.caps_seq_re.finditer(t_clean):
            cand = m.group(0).strip()
            if not cand:
                continue
            words = cand.split()
            if any(w in bad for w in words):
                continue
            if len(words) == 1 and len(words[0]) < 4:
                continue
            out.append(cand)

        out.append(t_clean)

        seen = set()
        uniq: List[str] = []
        for x in out:
            x = str(x).strip()
            if x and x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    # ---------- extraction ----------

    def _extract_row_fields(self, row: pd.Series, scan_cols: List[str]) -> Dict[str, Any]:
        tokens: List[str] = []
        for c in scan_cols:
            tokens.extend(self._split_tokens(row.get(c)))
        tokens = [t for t in tokens if not self._is_noise_token(t)]

        inv_group = None
        warehouse = None
        platform = None

        nomen_candidates: List[Tuple[int, str]] = []
        cp_candidates: List[Tuple[int, str]] = []

        for tok in tokens:
            if platform is None and self.platform_re.match(tok):
                platform = tok
                continue
            if inv_group is None and self.inv_group_re.match(tok):
                inv_group = tok
                continue
            if warehouse is None and self.warehouse_re.match(tok):
                warehouse = tok
                continue

            ns = self._score_nomenclature(tok)
            if ns >= 8:
                nomen_candidates.append((ns, self._normalize_nomenclature(tok)))

            for cand in self._counterparty_subtokens(tok):
                cs = self._score_counterparty(cand)
                if cs >= 10:
                    cp_candidates.append((cs, cand.strip()))

        counterparty = None
        if cp_candidates:
            cp_candidates.sort(key=lambda x: (-x[0], -len(x[1])))
            counterparty = cp_candidates[0][1]

        nomen = None
        if nomen_candidates:
            nomen_candidates.sort(key=lambda x: (-x[0], len(x[1])))
            nomen = nomen_candidates[0][1]

        if nomen and (self._looks_like_doc_header(nomen) or self._contains_doc_type(nomen)):
            nomen = None

        return {
            "Площадка": platform,
            "Готовая продукция/Товары": inv_group,
            "Склад": warehouse,
            "Номенклатура": nomen,
            "Контрагент": counterparty,
        }

    # ---------- main ----------

    def run(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = pd.read_csv(self.input_csv_path, encoding="utf-8-sig", dtype="string")
        df["_row_id"] = np.arange(len(df))

        df[self.debit_col] = df[self.debit_col].astype("string").str.strip()
        df[self.credit_col] = df[self.credit_col].astype("string").str.strip()

        doc_col = self._detect_doc_text_col(df)
        doc_text = df[doc_col].astype("string").str.strip()

        # ДокументКлюч = полный текст документа (с хвостом "от ... ...")
        df["ДокументКлюч"] = doc_text

        header_re = self._doc_header_re()
        hdr = doc_text.str.extract(header_re)

        # Берём только строки по нашим типам документов
        df_docs = df[hdr["dt"].notna()].copy()
        if df_docs.empty:
            raise ValueError("Не найдено строк с нужными документами (doc_types).")

        # ---- суммы по корреспонденциям ----
        amt = self._to_amount(df_docs[self.amount_col])
        debit = df_docs[self.debit_col]
        credit = df_docs[self.credit_col]

        is_rev = debit.str.startswith(self.revenue_debit_prefix, na=False) & credit.str.startswith(self.revenue_credit_prefix, na=False)
        is_vat = debit.str.startswith(self.vat_debit_prefix, na=False) & self._startswith_any(credit, self.vat_credit_prefixes)
        is_cogs = debit.str.startswith(self.cogs_debit_prefix, na=False) & self._startswith_any(credit, self.cogs_credit_prefixes)

        df_docs["__Выручка"] = np.where(is_rev.to_numpy(), amt.to_numpy(), 0.0)
        df_docs["__НДС_raw"] = np.where(is_vat.to_numpy(), amt.to_numpy(), 0.0)
        df_docs["__Себестоимость_raw"] = np.where(is_cogs.to_numpy(), amt.to_numpy(), 0.0)

        # фильтр документов: оставляем только те, где есть хоть одна из трёх сумм
        doc_tot = df_docs.groupby("ДокументКлюч", as_index=False)[["__Выручка", "__НДС_raw", "__Себестоимость_raw"]].sum()
        keep = set(
            doc_tot[
                (doc_tot["__Выручка"] != 0)
                | (doc_tot["__НДС_raw"] != 0)
                | (doc_tot["__Себестоимость_raw"] != 0)
            ]["ДокументКлюч"].tolist()
        )
        df_docs = df_docs[df_docs["ДокументКлюч"].isin(keep)].copy()

        # ---- извлечение аналитик ----
        # важно: НЕ сканируем doc_col, иначе заголовок документа лезет в номенклатуру
        exclude_scan = {
            self.debit_col, self.credit_col, self.amount_col, "_row_id",
            "__Выручка", "__НДС_raw", "__Себестоимость_raw",
            "ДокументКлюч", doc_col
        }
        scan_cols = [c for c in df_docs.columns if c not in exclude_scan]

        fields = df_docs.apply(lambda r: self._extract_row_fields(r, scan_cols), axis=1, result_type="expand")
        df_docs = pd.concat([df_docs, fields], axis=1)

        # doc-level аналитика (первое непустое)
        doc_info_cols = ["Площадка", "Контрагент", "Готовая продукция/Товары", "Склад"]
        if self.date_col in df_docs.columns:
            doc_info_cols = [self.date_col] + doc_info_cols

        doc_info = (
            df_docs.sort_values(["ДокументКлюч", "_row_id"])
            .groupby("ДокументКлюч", as_index=False)[doc_info_cols]
            .agg({c: self._pick_first_non_empty for c in doc_info_cols})
        )
        if self.date_col in doc_info.columns:
            doc_info = doc_info.rename(columns={self.date_col: "Дата"})
        else:
            doc_info["Дата"] = pd.NA

        # ---- группируем по ДокументКлюч + Номенклатура ----
        df_docs["__nom_key"] = df_docs["Номенклатура"].astype("string")
        df_docs["__nom_key"] = df_docs["__nom_key"].where(df_docs["__nom_key"].notna(), other="__NO_NOM__")

        agg = (
            df_docs.groupby(["ДокументКлюч", "__nom_key"], as_index=False)
            .agg({
                "__Выручка": "sum",
                "__НДС_raw": "sum",
                "__Себестоимость_raw": "sum",
                "Номенклатура": self._pick_first_non_empty,
                "Площадка": self._pick_first_non_empty,
                "Контрагент": self._pick_first_non_empty,
                "Готовая продукция/Товары": self._pick_first_non_empty,
                "Склад": self._pick_first_non_empty,
                self.date_col: self._pick_first_non_empty if self.date_col in df_docs.columns else "first",
            })
        )

        if self.date_col in agg.columns:
            agg = agg.rename(columns={self.date_col: "Дата"})
        else:
            agg["Дата"] = pd.NA

        # дозаполнение из doc_info
        agg = agg.merge(
            doc_info[["ДокументКлюч", "Дата", "Площадка", "Контрагент", "Готовая продукция/Товары", "Склад"]],
            on="ДокументКлюч",
            how="left",
            suffixes=("", "_doc"),
        )
        for col in ["Дата", "Площадка", "Контрагент", "Готовая продукция/Товары", "Склад"]:
            dc = f"{col}_doc"
            if dc in agg.columns:
                agg[col] = agg[col].fillna(agg[dc])
                agg = agg.drop(columns=[dc])

        # ---- Выручка ----
        agg["Выручка"] = pd.to_numeric(agg["__Выручка"], errors="coerce").fillna(0.0)

        # ---- НДС: суммарно по ДокументКлюч и распределяем по выручке внутри ДокументКлюч ----
        vat_total_by_doc = df_docs.groupby("ДокументКлюч")["__НДС_raw"].sum().astype(float)

        rev = agg["Выручка"].astype(float).fillna(0.0)
        rev_abs = rev.abs()
        agg["_rev_abs_w"] = np.where(rev != 0.0, rev_abs, 0.0)
        denom = agg.groupby("ДокументКлюч")["_rev_abs_w"].transform("sum").astype(float)

        vat_total_per_row = agg["ДокументКлюч"].map(vat_total_by_doc).astype(float).fillna(0.0)
        weight = np.where(denom.to_numpy() != 0.0, agg["_rev_abs_w"].to_numpy() / denom.to_numpy(), 0.0)
        agg["НДС"] = vat_total_per_row.to_numpy() * weight

        # если НДС есть, а выручки нет — кладём НДС в строку "__NO_NOM__" (или в первую строку документа)
        docs_no_rev = agg.loc[(denom == 0.0), "ДокументКлюч"].unique().tolist()
        if docs_no_rev:
            for dk in docs_no_rev:
                vat_val = float(vat_total_by_doc.get(dk, 0.0))
                if abs(vat_val) <= self.eps:
                    continue
                sub_idx = agg.index[agg["ДокументКлюч"] == dk].tolist()
                if not sub_idx:
                    continue
                idx_no_nom = agg.index[(agg["ДокументКлюч"] == dk) & (agg["__nom_key"] == "__NO_NOM__")].tolist()
                target_idx = idx_no_nom[0] if idx_no_nom else sub_idx[0]
                agg.loc[agg["ДокументКлюч"] == dk, "НДС"] = 0.0
                agg.loc[target_idx, "НДС"] = vat_val

        # ---- Себестоимость: НЕ распределяем. Если номенклатура не распознана — 0 ----
        is_real_nom = (agg["__nom_key"] != "__NO_NOM__")
        agg["Себестоимость"] = np.where(
            is_real_nom.to_numpy(),
            pd.to_numeric(agg["__Себестоимость_raw"], errors="coerce").fillna(0.0).to_numpy(),
            0.0
        )

        agg = agg.drop(columns=["_rev_abs_w"], errors="ignore")

        # ---- Заглушки ----
        # номенклатуру оставляем как "Без номенклатуры", НО потом выкинем строки с нулями
        agg["Номенклатура"] = agg["Номенклатура"].fillna("Без номенклатуры")
        agg["Контрагент"] = agg["Контрагент"].fillna("Без контрагента")

        # ---- итог ----
        out = agg[[
            "Дата",
            "ДокументКлюч",
            "Площадка",
            "Контрагент",
            "Номенклатура",
            "Готовая продукция/Товары",
            "Склад",
            "Выручка",
            "НДС",
            "Себестоимость",
        ]].copy()

        out["Дата"] = pd.to_datetime(out["Дата"], errors="coerce")
        for col in ["Выручка", "НДС", "Себестоимость"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

        # ---------- ВОТ НУЖНАЯ ПРАВКА ----------
        # Удаляем строки "Без номенклатуры", если по ним все суммы = 0
        bn = out["Номенклатура"].astype("string").str.strip().eq("Без номенклатуры")
        zeros = (out["Выручка"].abs() <= self.eps) & (out["НДС"].abs() <= self.eps) & (out["Себестоимость"].abs() <= self.eps)
        drop_mask = bn & zeros
        dropped = int(drop_mask.sum())
        out = out[~drop_mask].copy()
        # --------------------------------------

        # сохранить Excel
        with pd.ExcelWriter(self.output_excel_path, engine="openpyxl") as writer:
            out.to_excel(writer, sheet_name="result", index=False)

        # фильтр + заморозка
        try:
            from openpyxl import load_workbook
            wb = load_workbook(self.output_excel_path)
            ws = wb["result"]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            wb.save(self.output_excel_path)
        except Exception:
            pass

        meta = {
            "input_csv": self.input_csv_path,
            "output_excel": self.output_excel_path,
            "doc_col_used": doc_col,
            "rows_out": int(len(out)),
            "unique_docs": int(out["ДокументКлюч"].nunique()),
            "dropped_zero_no_nom_rows": dropped,
        }
        return out, meta


def main():
    input_csv = r"Z:\01 Администрация\Кобычев\МК P&L\report_transformed.csv"
    output_xlsx = r"Z:\01 Администрация\Кобычев\МК P&L\report_corr_docs_by_nomenclature.xlsx"

    job = CorrDocsByNomenclatureToExcel(input_csv_path=input_csv, output_excel_path=output_xlsx)
    out_df, meta = job.run()

    print("\n=== ГОТОВО: Excel сформирован ===")
    print("Excel:", meta["output_excel"])
    print("Колонка документа:", meta["doc_col_used"])
    print("Строк:", meta["rows_out"])
    print("Уникальных документов:", meta["unique_docs"])
    print("Удалено строк 'Без номенклатуры' с нулями:", meta["dropped_zero_no_nom_rows"])

    print("\n=== Preview (первые 30 строк) ===")
    with pd.option_context("display.max_columns", 25, "display.width", 240):
        print(out_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
