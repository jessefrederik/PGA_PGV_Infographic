#' @title Generate M3.4 PGV with Vs30 CSV (Surface Stations Only)
#'
#' @description
#' Combines M3.4 earthquake ground motion data from all four databases
#' (Groningen, Italy, Japan, California) into a single CSV, filtering out
#' borehole/downhole station recordings that would bias PGV comparisons.
#'
#' @details
#' Borehole sensors measure ground motion at depth, missing near-surface
#' site amplification, producing systematically lower PGV values than
#' surface stations.
#'
#' Filtering applied per dataset:
#' \itemize{
#'   \item Groningen: Keep surface flags only (G0, GS, B, OB). Exclude
#'     borehole flags G1-G4 (50-200m depth) and NL (no Vs30).
#'   \item Italy: Exclude WEL (borehole) housing type from sensor_location.csv.
#'     VAU (vault) stations are shallow pits, NOT deep boreholes — kept.
#'   \item Japan: K-NET (network 1): keep all except sensor_depth >= 9996
#'     (missing metadata). KiK-net (network 2): keep only depth == 0 (surface).
#'   \item California (CESMD): No filtering needed — CESMD excludes downhole
#'     recordings by policy.
#' }
#'
#' @author Jesse
#' @date March 2026
#'
#' @name generate_m34_csv
NULL

source(here::here("pipeline", "00_config.R"))
library(readxl)

# ==============================================================================
# GRONINGEN — Filter to surface stations via Flag column
# ==============================================================================

load_groningen_surface <- function() {
  message("\n", strrep("=", 70))
  message("Groningen: Loading surface stations only")
  message(strrep("=", 70))

  raw <- read_excel(DATA_PATHS$groningen, sheet = "GM")
  message("  Total records in Excel: ", nrow(raw))

  # Surface station flags (exclude G1-G4 borehole and NL reference)
  surface_flags <- c("G0", "GS", "B", "OB")

  message("  Flag distribution (all records):")
  print(table(raw$Flag, useNA = "ifany"))

  df <- raw %>%
    filter(Flag %in% surface_flags) %>%
    filter(ML >= MAG_MIN, ML <= MAG_MAX) %>%
    transmute(
      region = "Groningen",
      database = "GGMD",
      event_id = paste0("GRN_", EQID),
      station_id = as.character(STAT),
      magnitude = as.numeric(ML),
      depth_km = as.numeric(`Depth (km)`),
      epicentral_dist_km = as.numeric(`Repi (km)`),
      hypocentral_dist_km = sqrt(epicentral_dist_km^2 + depth_km^2),
      pgv_mms = as.numeric(PGV_GM) * 10,
      vs30_ms = as.numeric(`VS30 (m/s)`)
    ) %>%
    filter(!is.na(pgv_mms), pgv_mms > 0)

  message("  Surface M3.4 records: ", nrow(df))
  message("  Events: ", n_distinct(df$event_id))

  df
}

# ==============================================================================
# ITALY — Exclude WEL (borehole) housing type
# ==============================================================================

load_italy_surface <- function() {
  message("\n", strrep("=", 70))
  message("Italy: Excluding WEL (borehole) stations")
  message(strrep("=", 70))

  # Load sensor_location.csv to identify borehole stations
  sensor_file <- here("pipeline", "datafiles", "itaca_32_db", "sensor_location.csv")
  sensor_loc <- read_csv(
    sensor_file,
    col_names = FALSE,
    col_types = cols(.default = "c"),
    show_col_types = FALSE
  )

  # V1 = network, V2 = station_code, V9 = housing type
  borehole_stations <- sensor_loc %>%
    filter(X9 == "WEL") %>%
    transmute(station_id = paste(X1, X2, sep = ".")) %>%
    pull(station_id) %>%
    unique()

  message("  WEL (borehole) station IDs: ", paste(borehole_stations, collapse = ", "))

  # Load processed Italy data
  df <- read_csv(DATA_PATHS$italy, show_col_types = FALSE) %>%
    filter(
      magnitude >= MAG_MIN,
      magnitude <= MAG_MAX,
      mag_type == "ML"
    )

  n_before <- nrow(df)
  n_borehole <- sum(df$station_id %in% borehole_stations)
  message("  Records before filter: ", n_before)
  message("  Borehole records to remove: ", n_borehole)

  df <- df %>%
    filter(!station_id %in% borehole_stations) %>%
    transmute(
      region = "Italy",
      database = "ITACA",
      event_id = event_id,
      station_id = station_id,
      magnitude = magnitude,
      depth_km = depth_km,
      epicentral_dist_km = epicentral_dist_km,
      hypocentral_dist_km = sqrt(epicentral_dist_km^2 + depth_km^2),
      pgv_mms = pgv_mms,
      vs30_ms = vs30_ms
    ) %>%
    filter(!is.na(pgv_mms), pgv_mms > 0)

  message("  Surface M3.4 records: ", nrow(df))
  message("  Events: ", n_distinct(df$event_id))

  df
}

# ==============================================================================
# JAPAN — Filter K-NET (keep all, exclude depth >= 9996) and KiK-net (surface only)
# ==============================================================================

load_japan_surface <- function() {
  message("\n", strrep("=", 70))
  message("Japan: Filtering to surface stations")
  message(strrep("=", 70))

  # Load site_schema.tsv for network and depth info
  site_file <- here("pipeline", "datafiles", "flatfile-v2024", "site_schema.tsv")
  sites <- read_tsv(site_file, show_col_types = FALSE) %>%
    transmute(
      site_code = site_code,
      obs_network_id = as.integer(obs_network_id),
      sensor_depth = as.numeric(sensor_depth_glminus)
    )

  message("  Site schema loaded: ", nrow(sites), " stations")
  message("  K-NET (network 1): ", sum(sites$obs_network_id == 1))
  message("  KiK-net (network 2): ", sum(sites$obs_network_id == 2))

  # Surface station filter:
  # K-NET: keep all except sensor_depth >= 9996 (missing/unknown metadata)
  # KiK-net: keep only depth == 0 (surface sensors)
  surface_stations <- sites %>%
    filter(
      (obs_network_id == 1 & (is.na(sensor_depth) | sensor_depth < 9996)) |
      (obs_network_id == 2 & sensor_depth == 0)
    ) %>%
    pull(site_code)

  knet_excluded <- sites %>%
    filter(obs_network_id == 1, !is.na(sensor_depth), sensor_depth >= 9996) %>%
    nrow()
  kiknet_borehole <- sites %>%
    filter(obs_network_id == 2, sensor_depth != 0 | is.na(sensor_depth)) %>%
    nrow()

  message("  K-NET stations excluded (depth >= 9996): ", knet_excluded)
  message("  KiK-net borehole stations excluded: ", kiknet_borehole)
  message("  Surface stations kept: ", length(surface_stations))

  # Load processed Japan data
  df <- read_csv(DATA_PATHS$japan, show_col_types = FALSE) %>%
    filter(magnitude >= MAG_MIN, magnitude <= MAG_MAX,
           mag_type == "JMA")

  n_before <- nrow(df)
  message("  Records before filter: ", n_before)

  df <- df %>%
    filter(station_id %in% surface_stations) %>%
    transmute(
      region = "Japan",
      database = "J-SHIS",
      event_id = as.character(event_id),
      station_id = station_id,
      magnitude = magnitude,
      depth_km = depth_km,
      epicentral_dist_km = epicentral_dist_km,
      hypocentral_dist_km = sqrt(epicentral_dist_km^2 + depth_km^2),
      pgv_mms = pgv_mms,
      vs30_ms = vs30_ms
    ) %>%
    filter(!is.na(pgv_mms), pgv_mms > 0)

  message("  Surface M3.4 records: ", nrow(df))
  message("  Events: ", n_distinct(df$event_id))

  df
}

# ==============================================================================
# CALIFORNIA — No borehole filtering needed
# ==============================================================================

load_california_surface <- function() {
  message("\n", strrep("=", 70))
  message("California: No borehole filtering needed (CESMD policy)")
  message(strrep("=", 70))

  df <- read_csv(DATA_PATHS$california, show_col_types = FALSE) %>%
    filter(
      magnitude >= MAG_MIN,
      magnitude <= MAG_MAX,
      mag_type == "ML"
    ) %>%
    transmute(
      region = "California",
      database = "CESMD",
      event_id = event_id,
      station_id = station_id,
      magnitude = magnitude,
      depth_km = depth_km,
      epicentral_dist_km = epicentral_dist_km,
      hypocentral_dist_km = sqrt(epicentral_dist_km^2 + depth_km^2),
      pgv_mms = pgv_mms,
      vs30_ms = vs30_ms
    ) %>%
    filter(!is.na(pgv_mms), pgv_mms > 0)

  message("  Records: ", nrow(df))
  message("  Events: ", n_distinct(df$event_id))

  df
}

# ==============================================================================
# MAIN: Combine and write CSV
# ==============================================================================

generate_m34_csv <- function() {
  message("\n")
  message(strrep("=", 70))
  message("Generating m34_pgv_with_vs30.csv (Surface Stations Only)")
  message(strrep("=", 70))

  gron <- load_groningen_surface()
  italy <- load_italy_surface()
  japan <- load_japan_surface()
  california <- load_california_surface()

  combined <- bind_rows(gron, italy, japan, california)

  # Apply depth filter
  n_before_depth <- nrow(combined)
  n_deep <- sum(combined$depth_km > DEPTH_MAX_KM, na.rm = TRUE)
  n_na_depth <- sum(is.na(combined$depth_km))
  combined <- combined %>%
    filter(depth_km <= DEPTH_MAX_KM | is.na(depth_km))
  message("\nDepth filter (<= ", DEPTH_MAX_KM, " km):")
  message("  Removed ", n_deep, " deep records")
  if (n_na_depth > 0) message("  Kept ", n_na_depth, " records with unknown depth")

  # Summary
  message("\n", strrep("=", 70))
  message("Combined Dataset Summary")
  message(strrep("=", 70))
  message("Total records: ", nrow(combined))
  message("\nRecords per region:")
  print(table(combined$region))
  message("\nEvents per region:")
  event_summary <- combined %>%
    group_by(region) %>%
    summarise(
      n_records = n(),
      n_events = n_distinct(event_id),
      vs30_coverage = paste0(sum(!is.na(vs30_ms)), "/", n(), " (",
                             round(100 * sum(!is.na(vs30_ms)) / n()), "%)"),
      .groups = "drop"
    )
  print(event_summary)

  # Write output
  output_file <- here("data", "m34_pgv_with_vs30.csv")
  write_csv(combined, output_file)
  message("\nSaved to: ", output_file)

  invisible(combined)
}

# ==============================================================================
# RUN
# ==============================================================================

if (sys.nframe() == 0) {
  generate_m34_csv()
}
