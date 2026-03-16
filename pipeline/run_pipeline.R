#!/usr/bin/env Rscript
#' @title Run Full Data Pipeline
#'
#' @description
#' Executes the complete data pipeline from raw data extraction through
#' to the final m34_pgv_with_vs30.csv used by prepare_data.py.
#'
#' Execution order:
#'   1. Process individual country datasets (Layer 1 — can run in parallel)
#'   2. Generate combined M3.4 CSV with surface-station filtering (Layer 2)
#'
#' Prerequisites:
#'   Place raw data files in pipeline/datafiles/ (see DATA_SOURCES.md)
#'
#' Usage:
#'   Rscript pipeline/run_pipeline.R           # Run full pipeline
#'   Rscript pipeline/run_pipeline.R --skip-etl # Skip Layer 1, regenerate combined CSV only

library(here)

args <- commandArgs(trailingOnly = TRUE)
skip_etl <- "--skip-etl" %in% args

message("\n", strrep("=", 70))
message("PGV Comparison Pipeline")
message(strrep("=", 70))
message("Project root: ", here())
message("Skip ETL: ", skip_etl)

# ── Layer 1: Extract & process per-country datasets ──────────────────────────
if (!skip_etl) {
  message("\n\n>>> LAYER 1: Processing individual country datasets <<<\n")

  source(here("pipeline", "data_processing", "01_process_groningen.R"))
  process_groningen()

  source(here("pipeline", "data_processing", "02_process_italy.R"))
  process_italy()

  source(here("pipeline", "data_processing", "03_process_japan.R"))
  process_japan()

  source(here("pipeline", "data_processing", "04_process_california.R"))
  process_california()
} else {
  message("\nSkipping Layer 1 (ETL). Using existing processed CSVs.")
}

# ── Layer 2: Combine into final CSV with surface-station filtering ───────────
message("\n\n>>> LAYER 2: Generating combined M3.4 CSV <<<\n")
source(here("pipeline", "03_generate_m34_csv.R"))
generate_m34_csv()

message("\n", strrep("=", 70))
message("Pipeline complete!")
message("Output: ", here("data", "m34_pgv_with_vs30.csv"))
message("\nNext step: python3 prepare_data.py")
message(strrep("=", 70))
