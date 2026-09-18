"""Explicit Excel surface schemas and auditable, unit-normalized observations."""
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


@dataclass(frozen=True)
class MarketDataSchema:
    quote_axis: str
    iv_unit: str
    r_unit: str
    q_unit: str
    day_count: str

    def __post_init__(self):
        if self.quote_axis not in ('forward_moneyness_percent', 'absolute_strike'):
            raise ValueError('unsupported quote_axis')
        for name in ('iv_unit', 'r_unit', 'q_unit'):
            if getattr(self, name) not in ('percentage_points', 'decimal'):
                raise ValueError(f'unsupported {name}')
        if self.day_count not in ('ACT/365F', 'ACT/360'):
            raise ValueError('unsupported day_count')

    @property
    def year_basis(self):
        return 365. if self.day_count == 'ACT/365F' else 360.


SPX_MID_SCHEMA = MarketDataSchema('forward_moneyness_percent',
    'percentage_points', 'percentage_points', 'percentage_points', 'ACT/365F')
GOOGL_MID_SCHEMA = MarketDataSchema('absolute_strike',
    'decimal', 'decimal', 'decimal', 'ACT/360')


@dataclass(frozen=True)
class RejectedObservation:
    workbook: str
    sheet: str
    excel_row: int
    original_header: object
    reason: str
    source_values: tuple


class MarketDataError(ValueError):
    def __init__(self, message, rejected=()):
        self.rejected = tuple(rejected)
        super().__init__(message)


@dataclass(frozen=True)
class LoadedSurface:
    workbook: str
    sheet: str
    valuation_date: pd.Timestamp
    S0: float
    schema: MarketDataSchema
    observations: pd.DataFrame
    rejected: tuple


def _number(value, name, *, positive=False):
    a = np.asarray(value)
    if a.ndim or a.dtype.kind not in 'iuf' or not np.isfinite(a):
        raise ValueError(f'{name} must be a finite real number')
    value = float(a)
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def _date(value, name):
    if not isinstance(value, (date, datetime, str, pd.Timestamp)):
        raise ValueError(f'{name} must be a date')
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'invalid {name}') from exc
    if pd.isna(result) or result.tz is not None or result != result.normalize():
        raise ValueError(f'{name} must be a timezone-free calendar date')
    return result


def _unit(value, unit, name, *, positive=False):
    return _number(value, name, positive=positive)/(100. if unit == 'percentage_points' else 1.)


def load_surface(workbook, sheet, schema):
    """Read stored numeric headers verbatim; no magnitude-based unit inference.

    Ambiguous workbook metadata fails the load. Invalid individual quote/row
    values are returned as rejection records with the original source row.
    Fully blank formatted rows/columns are not observations.
    """
    if not isinstance(schema, MarketDataSchema):
        raise TypeError('schema must be MarketDataSchema')
    workbook = str(Path(workbook).resolve())
    book = load_workbook(workbook, read_only=True, data_only=False)
    try:
        rows = list(book[sheet].values)
    finally:
        book.close()
    if not rows:
        raise MarketDataError('empty worksheet')
    headers = rows[0]
    required = ('Act Date', 'Spot', 'Expiry', 'Exp Date', 'Risk Free', 'Impl (Yld)', 'ImplFwd')
    if any(headers.count(name) != 1 for name in required):
        raise MarketDataError('required metadata columns must occur exactly once')
    positions = {name: headers.index(name) for name in required}
    quote_columns = [(i, h) for i, h in enumerate(headers) if h is not None and h not in required]
    if not quote_columns or len({h for _, h in quote_columns}) != len(quote_columns):
        raise MarketDataError('quote headers must be nonempty and unique')
    if any(row[i] is not None for row in rows[1:] for i, h in enumerate(headers) if h is None):
        raise MarketDataError('populated cells under a missing header')
    data_rows = [(i, row) for i, row in enumerate(rows[1:], 2) if any(v is not None for v in row)]
    spots = {_number(row[positions['Spot']], 'Spot', positive=True)
             for _, row in data_rows if row[positions['Spot']] is not None}
    dates = {_date(row[positions['Act Date']], 'Act Date')
             for _, row in data_rows if row[positions['Act Date']] is not None}
    if len(spots) != 1 or len(dates) != 1:
        raise MarketDataError('exactly one spot and valuation date are required')
    spot, valuation = next(iter(spots)), next(iter(dates))
    records, rejected = [], []
    for excel_row, row in data_rows:
        raw = {name: row[index] for name, index in positions.items()}
        for column, header in quote_columns:
            source = dict(workbook=workbook, sheet=sheet, excel_row=excel_row,
                          original_header=header, source_values=tuple(zip(headers, row)))
            try:
                expiry = raw['Expiry']
                if not isinstance(expiry, str) or not expiry.strip():
                    raise ValueError('Expiry must be a nonempty label')
                expiration = _date(raw['Exp Date'], 'Exp Date')
                r = _unit(raw['Risk Free'], schema.r_unit, 'Risk Free')
                q = _unit(raw['Impl (Yld)'], schema.q_unit, 'Impl (Yld)')
                forward = _number(raw['ImplFwd'], 'ImplFwd', positive=True)
                iv = _unit(row[column], schema.iv_unit, 'IV', positive=True)
                if schema.quote_axis == 'absolute_strike':
                    strike = _number(header, 'strike header', positive=True)
                    moneyness = strike/forward
                else:
                    if not isinstance(header, str) or not header.endswith('%'):
                        raise ValueError('forward-moneyness header must end in %')
                    moneyness = _number(float(header[:-1])/100., 'moneyness', positive=True)
                    strike = moneyness*forward
                if not np.isfinite(strike) or not np.isfinite(moneyness) or strike <= 0 or moneyness <= 0:
                    raise ValueError('invalid derived strike/moneyness')
                records.append(dict(source, valuation_date=valuation, expiry=expiry,
                    expiration_date=expiration, day_count=schema.day_count,
                    S0=spot, r=r, q=q, F=forward, K=strike,
                    moneyness=moneyness, market_iv=iv))
            except (ValueError, TypeError, OverflowError) as exc:
                rejected.append(RejectedObservation(**source, reason=str(exc)))
    return LoadedSurface(workbook, sheet, valuation, spot, schema,
                         pd.DataFrame.from_records(records), tuple(rejected))
