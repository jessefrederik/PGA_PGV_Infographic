# Data Sources

Raw data files go in `pipeline/datafiles/`. They are gitignored due to size.

## Groningen (KNMI/NAM GGMD)

- **File**: `datafiles/groningenGMMdata.xlsx`
- **Source**: KNMI/NAM Groningen Ground Motion Database
- **Download**: https://public.yoda.uu.nl/geo/UU01/
- **Sheet used**: "GM" (geometric mean of horizontal components)
- **Magnitude**: ML (local magnitude)
- **PGV**: Geometric mean, pre-computed by KNMI (`PGV_GM` column, cm/s)

## Italy (ITACA 3.2)

- **Files**: `datafiles/itaca_32_db/` directory containing:
  - `event_origin.csv`, `event_magnitude.csv`, `station.csv`
  - `waveform_component_proc.csv`, `log_vs.csv`, `sensor_location.csv`
- **Source**: Italian Accelerometric Archive (ITACA 3.2)
- **Download**: http://itaca.mi.ingv.it/ItacaNet_32/#/home → "Database dump (CSV)"
- **Magnitude**: ML (filtered in pipeline)
- **PGV**: Geometric mean, computed from E/N components: `sqrt(PGV_E * PGV_N)`

## Japan (J-SHIS Flatfile 2024)

- **Files**: `datafiles/flatfile-v2024/` directory containing:
  - `source_schema.tsv` (27 MB — event catalog)
  - `site_schema.tsv` (400 KB — station metadata)
  - `smrec_schema.tsv` (5 GB — strong motion records)
- **Source**: NIED J-SHIS Strong Ground Motion Flat File 2024
- **DOI**: 10.17598/NIED.0032
- **Download**: https://www.j-shis.bosai.go.jp/en/labs/ground-motion-flatfile/ (free registration required)
- **Magnitude**: JMA (approximately equivalent to ML for M3-4 range)
- **PGV**: Geometric mean, computed from H1/H2 components: `sqrt(maxvel1 * maxvel2)`

## California (CESMD)

- **Source**: Center for Engineering Strong Motion Data API
- **API**: https://www.strongmotioncenter.org/wserv/
- **No manual download needed** — fetched automatically by `04_process_california.R`
- **Magnitude**: ML (filtered in pipeline)
- **PGV**: Geometric mean (verified against per-channel waveforms from SCEDC FDSN)

## Pipeline execution order

```
1. Rscript pipeline/run_pipeline.R        # Process raw data → combined CSV
2. python3 prepare_data.py                 # Combined CSV → scroll_data.json
```

Or skip Layer 1 if processed CSVs already exist:
```
Rscript pipeline/run_pipeline.R --skip-etl
```
