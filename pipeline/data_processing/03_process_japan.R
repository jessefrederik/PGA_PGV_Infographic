#' @title Process J-SHIS (Japan) Strong Ground Motion Flatfile
#'
#' @description
#' Processes the NIED J-SHIS Strong Ground Motion Flat File 2024 to extract
#' M3.4-3.7 earthquake ground motion data from Japan.
#'
#' @details
#' Data source: NIED J-SHIS Strong Ground Motion Flat File 2024
#' DOI: 10.17598/NIED.0032
#' Download: https://www.j-shis.bosai.go.jp/en/labs/ground-motion-flatfile/
#'   Registration required (free). Download the flatfile package.
#'
#' Files required (place in datafiles/flatfile-v2024/):
#' \itemize{
#'   \item source_schema.tsv (27 MB) - Event catalog with magnitudes
#'   \item site_schema.tsv (400 KB) - Station metadata with Vs30
#'   \item smrec_schema.tsv (5 GB) - Strong motion records
#' }
#'
#' PGV definition: Geometric mean of horizontal components
#' \deqn{PGV_{GM} = \sqrt{maxvel1 \times maxvel2}}
#' where maxvel1 = NS component, maxvel2 = EW component
#'
#' Magnitude: JMA magnitude (approximately equivalent to ML for M3-4 range)
#'
#' @note The smrec_schema.tsv file is approximately 5 GB. Processing requires
#'   sufficient RAM. The script uses data.table for efficient loading.
#'
#' @references
#' NIED (2024). Strong Ground Motion Flat File. DOI: 10.17598/NIED.0032
#'
#' @author Jesse
#' @date January 2025
#'
#' @name process_japan
NULL

library(tidyverse)
library(data.table)
library(here)

# ==============================================================================
# Configuration
# ==============================================================================

#' @title Path to J-SHIS flatfile directory
INPUT_DIR <- here("pipeline", "datafiles", "flatfile-v2024")

#' @title Path to output processed CSV
OUTPUT_FILE <- here("pipeline", "data", "processed", "japan_jshis_pgv.csv")

#' @title Minimum magnitude threshold
#' @description Set slightly below target range (3.35-3.45) to ensure all
#'   events are captured. Final filtering happens in analysis scripts.
MAG_MIN <- 3.3

#' @title Maximum magnitude threshold
MAG_MAX <- 3.5

#' @title Maximum depth for shallow crustal events (km)
MAX_DEPTH_KM <- 15

#' @title Minimum stations per event for inclusion
MIN_STATIONS <- 5

# ==============================================================================
# Haversine Distance Function
# ==============================================================================

#' Calculate Haversine distance between two points
#'
#' @description
#' Computes the great-circle distance between two geographic coordinates
#' using the Haversine formula.
#'
#' @param lat1 Latitude of first point (degrees)
#' @param lon1 Longitude of first point (degrees)
#' @param lat2 Latitude of second point (degrees)
#' @param lon2 Longitude of second point (degrees)
#'
#' @return Distance in kilometers
#'
#' @details
#' Uses Earth radius of 6371 km.
#'
#' @examples
#' # Distance from Tokyo to Osaka
#' haversine_km(35.7, 139.7, 34.7, 135.5)  # ~400 km
#'
#' @export
haversine_km <- function(lat1, lon1, lat2, lon2) {
  to_rad <- pi / 180
  lat1 <- lat1 * to_rad; lat2 <- lat2 * to_rad
  lon1 <- lon1 * to_rad; lon2 <- lon2 * to_rad
  dlat <- lat2 - lat1; dlon <- lon2 - lon1
  a <- sin(dlat/2)^2 + cos(lat1) * cos(lat2) * sin(dlon/2)^2
  6371 * 2 * asin(sqrt(a))
}

# ==============================================================================
# Processing Function
# ==============================================================================

#' Process J-SHIS flatfile to standardized format
#'
#' @description
#' Reads the J-SHIS TSV flatfile and extracts M3.4-3.7 events with
#' standardized column names compatible with the ML comparison pipeline.
#'
#' @return Invisibly returns a tibble with processed data.
#'   Also writes the data to \code{OUTPUT_FILE}.
#'
#' @details
#' Processing steps:
#' \enumerate{
#'   \item Load event catalog (source_schema.tsv)
#'   \item Filter to magnitude range and shallow depth
#'   \item Load station catalog (site_schema.tsv)
#'   \item Load strong motion records (smrec_schema.tsv) - large file
#'   \item Compute geometric mean PGV from horizontal components
#'   \item Join all tables
#'   \item Compute epicentral distances via Haversine
#'   \item Filter to events with adequate station coverage
#' }
#'
#' Output columns (standardized schema):
#' \describe{
#'   \item{database}{Always "Japan (J-SHIS)"}
#'   \item{event_id}{Event source ID}
#'   \item{event_date}{Event datetime as character}
#'   \item{magnitude}{JMA magnitude}
#'   \item{mag_type}{Always "JMA"}
#'   \item{depth_km}{Hypocentral depth (km)}
#'   \item{station_id}{Station code}
#'   \item{epicentral_dist_km}{Epicentral distance (km)}
#'   \item{pgv_mms}{PGV geometric mean (mm/s)}
#'   \item{pga_g}{PGA geometric mean (g)}
#'   \item{vs30_ms}{Vs30 at station (m/s)}
#' }
#'
#' @section Memory Requirements:
#' The smrec_schema.tsv file is approximately 5 GB. Processing uses
#' data.table::fread() with column selection to minimize memory usage,
#' but at least 8 GB RAM is recommended.
#'
#' @examples
#' \dontrun{
#' df <- process_japan()
#' summary(df$epicentral_dist_km)
#' }
#'
#' @export
process_japan <- function() {

  message("=" |> strrep(70))
  message("Processing J-SHIS (Japan) Strong Ground Motion Flatfile")
  message("=" |> strrep(70))

  if (!dir.exists(INPUT_DIR)) {
    stop("Input directory not found: ", INPUT_DIR, "\n",
         "Please download J-SHIS flatfile from:\n",
         "https://www.j-shis.bosai.go.jp/en/labs/ground-motion-flatfile/")
  }

  # Load event catalog
  message("\nLoading event catalog...")
  sources <- fread(file.path(INPUT_DIR, "source_schema.tsv"),
                   sep = "\t", header = TRUE, data.table = FALSE) %>%
    as_tibble() %>%
    mutate(
      magnitude = as.numeric(mjma),
      mag_type = "JMA",
      depth_km = as.numeric(jem_depth)
    ) %>%
    filter(
      !is.na(magnitude),
      magnitude >= MAG_MIN - 0.1,  # Slightly wider for initial filter
      magnitude <= MAG_MAX + 0.1,
      !is.na(depth_km),
      depth_km <= MAX_DEPTH_KM
    ) %>%
    select(eq_source_id, event_date = jem_origin_time,
           event_lat = jem_lat, event_lon = jem_lon, depth_km, magnitude, mag_type)

  message("  Events in magnitude range: ", nrow(sources))

  # Load station catalog
  message("Loading station catalog...")
  sites <- fread(file.path(INPUT_DIR, "site_schema.tsv"),
                 sep = "\t", header = TRUE, data.table = FALSE) %>%
    as_tibble() %>%
    transmute(
      siteid2 = siteid2,
      station_code = site_code,
      sta_lat = as.numeric(lat),
      sta_lon = as.numeric(lon),
      vs30_ms = as.numeric(vs30)
    ) %>%
    filter(!is.na(sta_lat), !is.na(sta_lon))
  message("  Stations: ", nrow(sites))

  # Load strong motion records (large file - 5GB)
  message("Loading strong motion records (this may take a while)...")
  records_file <- file.path(INPUT_DIR, "smrec_schema.tsv")
  message("  File size: ", round(file.info(records_file)$size / 1e9, 2), " GB")

  records <- fread(records_file, sep = "\t", header = TRUE,
                   select = c("site_id", "eq_source_id", "maxvel0", "maxvel1",
                              "maxvel2", "maxacc0", "maxacc1", "maxacc2"),
                   showProgress = TRUE, data.table = FALSE) %>%
    as_tibble() %>%
    filter(eq_source_id %in% sources$eq_source_id)

  message("  Records for target events: ", nrow(records))

  # Compute geometric mean PGV from horizontal components
  message("Computing geometric mean PGV...")
  records <- records %>%
    mutate(
      siteid2 = as.integer(floor(site_id / 10)),
      pgv_h1 = as.numeric(maxvel1),
      pgv_h2 = as.numeric(maxvel2),
      pgv_mms = sqrt(pgv_h1 * pgv_h2) * 10,  # Convert cm/s to mm/s
      pga_h1 = as.numeric(maxacc1),
      pga_h2 = as.numeric(maxacc2),
      pga_g = sqrt(pga_h1 * pga_h2) / 980.665
    ) %>%
    filter(!is.na(pgv_mms), pgv_mms > 0)

  # Join with events and sites
  message("Joining tables...")
  combined <- records %>%
    inner_join(sources, by = "eq_source_id") %>%
    inner_join(sites, by = "siteid2") %>%
    mutate(
      epicentral_dist_km = haversine_km(event_lat, event_lon, sta_lat, sta_lon)
    ) %>%
    filter(!is.na(epicentral_dist_km))

  # Select events with good coverage
  message("Selecting events with good station coverage...")
  event_stats <- combined %>%
    group_by(eq_source_id) %>%
    summarise(
      n_stations = n(),
      n_near = sum(epicentral_dist_km <= 20),
      .groups = "drop"
    ) %>%
    filter(n_stations >= MIN_STATIONS, n_near >= 2)

  combined <- combined %>%
    filter(eq_source_id %in% event_stats$eq_source_id)

  message("  Events with good coverage: ", n_distinct(combined$eq_source_id))

  # Final output
  processed <- combined %>%
    filter(magnitude >= MAG_MIN, magnitude <= MAG_MAX) %>%
    transmute(
      database = "Japan (J-SHIS)",
      event_id = as.character(eq_source_id),
      event_date = as.character(event_date),
      magnitude = magnitude,
      mag_type = mag_type,
      depth_km = depth_km,
      station_id = station_code,
      epicentral_dist_km = epicentral_dist_km,
      pgv_mms = pgv_mms,
      pga_g = pga_g,
      vs30_ms = vs30_ms
    )

  # Summary
  message("\n--- Processed Dataset ---")
  message("Events: ", n_distinct(processed$event_id))
  message("Records: ", nrow(processed))
  message("Vs30 coverage: ", sum(!is.na(processed$vs30_ms)), " / ", nrow(processed),
          " (", round(100 * sum(!is.na(processed$vs30_ms)) / nrow(processed)), "%)")

  # Save
  dir.create(dirname(OUTPUT_FILE), recursive = TRUE, showWarnings = FALSE)
  write_csv(processed, OUTPUT_FILE)
  message("\nSaved to: ", OUTPUT_FILE)

  invisible(processed)
}

# ==============================================================================
# Run
# ==============================================================================

if (sys.nframe() == 0) {
  process_japan()
}
