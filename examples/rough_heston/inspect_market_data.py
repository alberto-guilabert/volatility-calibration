"""Phase-7A ingestion smoke check only: no optimizer or market calibration."""
from pathlib import Path

from volcal.market_data.preprocessing import DataLoader
from volcal.market_data.schema import SPX_MID_SCHEMA, GOOGL_MID_SCHEMA
from volcal.market_data.adapter import prepare_quotes


def main():
    root = Path(__file__).resolve().parents[2]
    datasets = [('SPX', 'spx/SPX_17_10_25.xlsx', SPX_MID_SCHEMA),
                ('GOOGL', 'googl/GOOGL_16_12_25.xlsx', GOOGL_MID_SCHEMA)]
    for name, relative, schema in datasets:
        path = root/'data'/relative
        result = prepare_quotes(DataLoader(path.parent, path.name).load_surface(schema, 'Mid'))
        frame = result.observations
        first = frame.sort_values('T').iloc[0]
        print(f'{name} Mid | day count: {schema.day_count} | valuation: {first.valuation_date.date()}')
        print(f'  quotes: {len(frame)} | expiries: {frame.expiration_date.nunique()} | rejected: {len(result.rejected)}')
        print(f'  T: [{frame["T"].min():.12g}, {frame["T"].max():.12g}]')
        print(f'  K/F: [{frame.moneyness.min():.12g}, {frame.moneyness.max():.12g}]')
        print(f'  first maturity: {first.expiry} ({first.expiration_date.date()}) '
              f'S0={first.S0:.15g}, r={first.r:.15g}, q={first.q:.15g}, F={first.F:.15g}')
        print(f'  maximum absolute forward residual: {frame.forward_absolute_residual.max():.15g}')
        print(f'  maximum absolute relative forward residual: {frame.forward_relative_residual.abs().max():.15g}')
        for rejected in result.rejected:
            print(f'  REJECTED row {rejected.excel_row}, header {rejected.original_header!r}: {rejected.reason}')


if __name__ == '__main__':
    main()
