"""Generic market surface -> forward-aware rough-Heston quote contract.

No calibration, curve estimation, bid/ask alignment or source-value repair.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from volcal.rough_heston.calibrator import PreparedQuotes
from volcal.utils import black
from .schema import LoadedSurface, MarketDataError, RejectedObservation


@dataclass(frozen=True)
class PreparedMarketData:
    quotes: PreparedQuotes
    source: LoadedSurface
    observations: pd.DataFrame
    rejected: tuple


def prepare_quotes(surface):
    """Return quotes, per-observation provenance and explicit rejection records.

    Residual sign is rates-derived minus supplied forward; relative residual
    divides by the supplied forward. Diagnostics never replace market inputs.
    All-rejected surfaces raise MarketDataError carrying the rejection records.
    """
    if not isinstance(surface, LoadedSurface):
        raise TypeError('surface must be LoadedSurface')
    records, rejected = [], list(surface.rejected)
    for row in surface.observations.to_dict('records'):
        try:
            t = (row['expiration_date']-row['valuation_date']).days/surface.schema.year_basis
            if t <= 0:
                raise ValueError('expiration must be after valuation date')
            typ = 'put' if row['K'] < row['F'] else 'call'
            args = dict(iv=row['market_iv'], T=t, K=row['K'], F=row['F'], r=row['r'])
            price = float(black.price(**args, option_type=typ))
            vega = float(black.vega(**args))
            if not np.isfinite(price) or not np.isfinite(vega) or price < 0 or vega < 0:
                raise ValueError('nonfinite or negative Black price/vega')
            # Diagnostic overflow is explicit, but must not reject a usable quote.
            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                rates_forward = row['S0']*np.exp((row['r']-row['q'])*t)
            residual = rates_forward-row['F']
            records.append(dict(row, T=t, option_type=typ, market_price=price,
                market_vega=vega, rates_forward=rates_forward,
                forward_residual=residual, forward_absolute_residual=abs(residual),
                forward_relative_residual=residual/row['F'],
                forward_diagnostic_status='ok' if np.isfinite(rates_forward) and rates_forward > 0
                                          else 'nonfinite or underflowed rates forward'))
        except (ValueError, FloatingPointError, OverflowError) as exc:
            source = {name: row[name] for name in ('workbook', 'sheet', 'excel_row',
                                                  'original_header', 'source_values')}
            rejected.append(RejectedObservation(**source, reason=str(exc)))
    if not records:
        raise MarketDataError(f'no usable quotes; {len(rejected)} rejected observations', rejected)
    observations = pd.DataFrame.from_records(records)
    fields = ('T', 'K', 'S0', 'r', 'q', 'F', 'option_type', 'market_price', 'market_iv', 'market_vega')
    quotes = PreparedQuotes(**{name: observations[name].to_numpy() for name in fields})
    return PreparedMarketData(quotes, surface, observations, tuple(rejected))
