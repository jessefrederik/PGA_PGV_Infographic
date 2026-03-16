#' @title Process ITACA 3.2 (Italy) Database
#'
#' @description
#' Processes the Italian Accelerometric Archive (ITACA 3.2) database dump
#' to extract M3.4-3.7 earthquake ground motion data.
#'
#' @details
#' Data source: ITACA 3.2 (Italian Accelerometric Archive)
#' Download: http://itaca.mi.ingv.it/ItacaNet_32/#/home
#'   Go to "Download" section, select "Database dump (CSV)"
#'
#' Files required (place in datafiles/itaca_32_db/):
#' \itemize{
#'   \item event_origin.csv - Event locations and depths
#'   \item event_magnitude.csv - Magnitude values
#'   \item station.csv - Station coordinates
#'   \item waveform_component_proc.csv - Component-level PGV/PGA
#'   \item log_vs.csv - Vs30 measurements (optional, for site correction)
#' }
#'
#' PGV definition: Geometric mean computed from E (East) and N (North) components
#' \deqn{PGV_{GM} = \sqrt{PGV_E \times PGV_N}}
#'
#' @references
#' Pacor et al. (2011). ITACA 1.0: a web portal for the dissemination
#' of the Italian strong motion data. Seismological Research Letters.
#'
#' @author Jesse
#' @date January 2025
#'
#' @name process_italy
NULL

library(tidyverse)
library(here)

# ==============================================================================
# Configuration
# ==============================================================================

#' @title Path to ITACA database dump directory
INPUT_DIR <- here("pipeline", "datafiles", "itaca_32_db")

#' @title Path to output processed CSV
OUTPUT_FILE <- here("pipeline", "data", "processed", "italy_itaca_pgv.csv")

#' @title Minimum magnitude threshold
#' @description Set slightly below target range (3.35-3.45) to ensure all
#'   events are captured. Final filtering happens in analysis scripts.
MAG_MIN <- 3.3

#' @title Maximum magnitude threshold
MAG_MAX <- 3.5

#' @title Maximum depth for shallow crustal events (km)
MAX_DEPTH_KM <- 15

# ==============================================================================
# Haversine Distance Function
# ==============================================================================

#' Calculate Haversine distance between two points
#'
#' @description
#' Computes the great-circle distance between two geographic coordinates
#' using the Haversine formula.
#'
#' @param lon1 Longitude of first point (degrees)
#' @param lat1 Latitude of first point (degrees)
#' @param lon2 Longitude of second point (degrees)
#' @param lat2 Latitude of second point (degrees)
#'
#' @return Distance in kilometers
#'
#' @details
#' Uses Earth radius of 6371 km.
#'
#' @examples
#' # Distance from Amsterdam to Rome
#' haversine_km(4.9, 52.4, 12.5, 41.9)  # ~1300 km
#'
#' @export
haversine_km <- function(lon1, lat1, lon2, lat2) {
  R <- 6371
  dLat <- (lat2 - lat1) * pi / 180
  dLon <- (lon2 - lon1) * pi / 180
  a <- sin(dLat / 2)^2 +
    cos(lat1 * pi / 180) * cos(lat2 * pi / 180) * sin(dLon / 2)^2
  c <- 2 * atan2(sqrt(a), sqrt(1 - a))
  R * c
}

# ==============================================================================
# Processing Function
# ==============================================================================

#' Process ITACA 3.2 database to standardized format
#'
#' @description
#' Reads the ITACA CSV database dump files and extracts M3.4-3.7 events
#' with standardized column names compatible with the ML comparison pipeline.
#'
#' @return Invisibly returns a tibble with processed data.
#'   Also writes the data to \code{OUTPUT_FILE}.
#'
#' @details
#' Processing steps:
#' \enumerate{
#'   \item Load event origins (filter to preferred solutions)
#'   \item Load magnitudes (filter to priority 1)
#'   \item Load station coordinates
#'   \item Load waveform components (E and N only)
#'   \item Compute geometric mean PGV from E/N pairs
#'   \item Join all tables
#'   \item Compute epicentral distances via Haversine
#'   \item Optionally join Vs30 from log_vs.csv
#' }
#'
#' Output columns (standardized schema):
#' \describe{
#'   \item{database}{Always "Italy (ITACA)"}
#'   \item{event_id}{Event identifier}
#'   \item{event_date}{Event datetime as character}
#'   \item{magnitude}{Local magnitude (ML)}
#'   \item{mag_type}{Magnitude type (filtered to ML)}
#'   \item{depth_km}{Hypocentral depth (km)}
#'   \item{station_id}{Station code as network.station_code}
#'   \item{epicentral_dist_km}{Epicentral distance (km)}
#'   \item{pgv_mms}{PGV geometric mean (mm/s)}
#'   \item{pga_g}{PGA geometric mean (g)}
#'   \item{vs30_ms}{Vs30 at station (m/s) if available}
#' }
#'
#' @section ITACA CSV Format:
#' The ITACA CSV files have no headers. Column names are defined in the
#' function based on the database schema documentation.
#'
#' @examples
#' \dontrun{
#' df <- process_italy()
#' table(df$mag_type)
#' }
#'
#' @export
process_italy <- function() {
  message("=" |> strrep(70))
  message("Processing ITACA 3.2 (Italy) Database")
  message("=" |> strrep(70))

  if (!dir.exists(INPUT_DIR)) {
    stop(
      "Input directory not found: ",
      INPUT_DIR,
      "\n",
      "Please download ITACA database dump from http://itaca.mi.ingv.it/"
    )
  }

  # Column definitions (ITACA CSVs have no headers)
  event_cols <- c(
    "event_id",
    "priority",
    "datetime_str",
    "time_unc",
    "latitude",
    "longitude",
    "depth_km",
    "lat_unc",
    "lon_unc",
    "depth_unc",
    "is_fixed",
    "ref_id",
    "is_preferred",
    "geom",
    "x15",
    "x16",
    "x17",
    "ts_modified"
  )

  mag_cols <- c(
    "event_id",
    "priority",
    "mag_type",
    "method",
    "ref_id",
    "magnitude",
    "uncertainty",
    "flag",
    "x9",
    "x10",
    "x11",
    "ts_modified"
  )

  station_cols <- c(
    "network",
    "station_code",
    "location",
    "nation",
    "name",
    "latitude",
    "longitude",
    "elevation",
    "install_date",
    "remove_date",
    "address",
    "vs30_flag",
    "description",
    "region1",
    "region2",
    "region3",
    "ts_modified",
    "is_visible",
    "is_reference",
    "is_approved",
    "ref_id",
    "geom",
    "ts_created",
    "created_by",
    "created_by2",
    "ts_modified2",
    "is_permanent",
    "x28"
  )

  component_cols <- c(
    "network",
    "station_code",
    "location",
    "instrument",
    "component",
    "event_id",
    "version",
    "npts",
    "dt",
    "lowcut",
    "highcut",
    "pga",
    "pgv",
    "pgd",
    "duration",
    "arias",
    "cav",
    "housner",
    "ts_modified"
  )

  # Load events
  message("\nLoading event origins...")
  events <- read_csv(
    file.path(INPUT_DIR, "event_origin.csv"),
    col_names = event_cols[1:18],
    col_types = cols(.default = "c"),
    show_col_types = FALSE
  ) %>%
    filter(is_preferred == "1" | priority == "1") %>%
    transmute(
      event_id = event_id,
      event_date = datetime_str,
      event_lat = as.numeric(latitude),
      event_lon = as.numeric(longitude),
      depth_km = as.numeric(depth_km)
    ) %>%
    distinct(event_id, .keep_all = TRUE)
  message("  Events: ", nrow(events))

  # Load magnitudes
  message("Loading magnitudes...")
  magnitudes <- read_csv(
    file.path(INPUT_DIR, "event_magnitude.csv"),
    col_names = mag_cols[1:12],
    col_types = cols(.default = "c"),
    show_col_types = FALSE
  ) %>%
    filter(priority == "1") %>%
    transmute(
      event_id = event_id,
      mag_type = toupper(mag_type),
      magnitude = as.numeric(magnitude)
    )
  message("  Magnitudes: ", nrow(magnitudes))

  # Load stations
  message("Loading stations...")
  stations <- read_csv(
    file.path(INPUT_DIR, "station.csv"),
    col_names = station_cols[1:28],
    col_types = cols(.default = "c"),
    show_col_types = FALSE
  ) %>%
    transmute(
      network = network,
      station_code = station_code,
      sta_lat = as.numeric(latitude),
      sta_lon = as.numeric(longitude)
    ) %>%
    distinct(network, station_code, .keep_all = TRUE)
  message("  Stations: ", nrow(stations))

  # Load component data and compute geometric mean
  message("Loading waveform components...")
  components <- read_csv(
    file.path(INPUT_DIR, "waveform_component_proc.csv"),
    col_names = component_cols[1:19],
    col_types = cols(.default = "c"),
    show_col_types = FALSE
  ) %>%
    filter(component %in% c("E", "N")) %>%
    transmute(
      network = network,
      station_code = station_code,
      event_id = event_id,
      component = component,
      pga = abs(as.numeric(pga)),
      pgv = abs(as.numeric(pgv))
    ) %>%
    filter(!is.na(pgv), pgv > 0)
  message("  Component records (E/N): ", nrow(components))

  # Compute geometric mean from E and N
  message("Computing geometric mean PGV from E/N components...")
  waveforms <- components %>%
    group_by(network, station_code, event_id) %>%
    filter(n() == 2) %>%
    summarise(
      pgv_mms = sqrt(prod(pgv)) * 10,  # Convert cm/s to mm/s
      pga_gal = sqrt(prod(pga)),
      .groups = "drop"
    )
  message("  Records with both components: ", nrow(waveforms))

  # Join all tables
  message("Joining tables...")
  result <- waveforms %>%
    inner_join(events, by = "event_id") %>%
    inner_join(magnitudes, by = "event_id") %>%
    inner_join(stations, by = c("network", "station_code")) %>%
    filter(
      !is.na(depth_km),
      depth_km >= 0,
      depth_km <= MAX_DEPTH_KM,
      magnitude >= MAG_MIN,
      magnitude <= MAG_MAX
    ) %>%
    mutate(
      epicentral_dist_km = haversine_km(sta_lon, sta_lat, event_lon, event_lat),
      station_id = paste(network, station_code, sep = "."),
      pga_g = pga_gal / 980.665
    )

  # Load Vs30 if available
  vs30_file <- file.path(INPUT_DIR, "log_vs.csv")
  if (file.exists(vs30_file)) {
    message("Loading Vs30 data...")
    vs30_cols <- c(
      "network",
      "station_code",
      "location",
      "x4",
      "x5",
      "lat",
      "lon",
      "x8",
      "x9",
      "method",
      "vs30",
      "x12",
      "x13",
      "x14",
      "x15",
      "x16",
      "ec8",
      "x18",
      "ec8_2",
      "ts_modified"
    )
    vs30_data <- read_csv(
      vs30_file,
      col_names = vs30_cols,
      col_types = cols(.default = "c"),
      show_col_types = FALSE
    ) %>%
      mutate(vs30 = as.numeric(vs30)) %>%
      filter(!is.na(vs30), vs30 > 0) %>%
      mutate(station_id = paste(network, station_code, sep = ".")) %>%
      select(station_id, vs30_ms = vs30) %>%
      distinct(station_id, .keep_all = TRUE)

    result <- result %>%
      left_join(vs30_data, by = "station_id")
    message(
      "  Vs30 coverage: ",
      sum(!is.na(result$vs30_ms)),
      " / ",
      nrow(result)
    )
  } else {
    result$vs30_ms <- NA_real_
  }

  # Final output
  processed <- result %>%
    transmute(
      database = "Italy (ITACA)",
      event_id = event_id,
      event_date = event_date,
      magnitude = magnitude,
      mag_type = mag_type,
      depth_km = depth_km,
      station_id = station_id,
      epicentral_dist_km = epicentral_dist_km,
      pgv_mms = pgv_mms,
      pga_g = pga_g,
      vs30_ms = vs30_ms
    ) %>%
    filter(!is.na(pgv_mms), !is.na(epicentral_dist_km))

  # Summary
  message("\n--- Processed Dataset ---")
  message("Events: ", n_distinct(processed$event_id))
  message("Records: ", nrow(processed))
  message(
    "Magnitude types: ",
    paste(unique(processed$mag_type), collapse = ", ")
  )

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
  process_italy()
}
