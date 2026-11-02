# Mind the Mic

Code for "Mind the Mic" (NeurIPS 2026 TAE workshop).

## Regenerate figures

```
pip install -r requirements.txt
python code/run.py m_figures
```

## Full reproduction

Needs Python 3.11 to 3.13, 16 GB RAM and 140 GB disk.

```
bash scripts/fetch_realimpact.sh
bash scripts/fetch_diffrir.sh
bash scripts/setup_diffrir.sh
bash scripts/reproduce_all.sh
```

CLAP values in `outputs/` are invalid (padding bug).

## Data

- [RealImpact](https://github.com/samuel-clarke/RealImpact)
- [DiffRIR dataset](https://doi.org/10.5281/zenodo.11195833) (CC BY 4.0)
- [DiffRIR code](https://github.com/maswang32/hearinganythinganywhere), commit `9fb1464`, source of the splits in `data_splits/` and the envelope metric in `m_env_band` and `m_env_perposition`

## Scripts

| Script | Paper |
|---|---|
| `m_within_column` | 0.1376 result, Table 4, Figure 3 |
| `m_cluster_boot` | interval [0.1158, 0.1634] |
| `m_df_bias_check` | Section 3.2 bias |
| `m_pathdiff` | 0.940 m path difference |
| `m_controls` | 0.0362 control, Table 4, Figure 3 |
| `m_snr` | Section 3.2, Figure 4 |
| `m_grid` | averaging and sample rate checks |
| `m_diffrir_final` | ratios 8.6 to 12.8 |
| `m_matched_displacement` | Table 1, ratios 7.9 to 10.1 |
| `m_perposition` | Tables 1 and 2 |
| `m_env_perposition`, `m_env_band` | Section 3.5 |
| `m_backup_controls` | Section 3.4 checks |
| `m_floor_verify` | Section 3.6, Figure 2 caption |
| `m_reproduce` | Appendix A factor of 5.2 |
| `m_figures` | Figures 1 to 4 |

## License

MIT for `code/` and `scripts/`. CC BY 4.0 for `cache/` and the receiver positions in `data_splits/`.
