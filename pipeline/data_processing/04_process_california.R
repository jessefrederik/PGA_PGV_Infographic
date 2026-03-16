#' @title Fetch and Process CESMD (California) Data
#'
#' @description
#' Fetches M3.4-3.7 earthquake ground motion data from California via the
#' CESMD (Center for Engineering Strong Motion Data) web service API.
#'
#' @details
#' Data source: Center for Engineering Strong Motion Data (CESMD)
#' API Documentation: https://www.strongmotioncenter.org/wserv/
#' No manual download required - data fetched automatically via API.
#'
#' PGV definition: Geometric mean (CESMD standard convention)
#'
#' The script:
#' \enumerate{
#'   \item Queries the CESMD events API for M3.4-3.7 earthquakes
#'   \item Fetches station records for each event
#'   \item Caches results locally (valid for 24 hours)
#'   \item Processes to standardized format
#' }
#'
#' @note API queries include rate limiting (0.3s delay between requests)
#'   to avoid overwhelming the server.
#'
#' @author Jesse
#' @date January 2025
#'
#' @name process_california
NULL

library(tidyverse)
library(httr2)
library(jsonlite)
library(here)

# ==============================================================================
# Configuration
# ==============================================================================

#' @title CESMD API base URL
CESMD_BASE <- "https://www.strongmotioncenter.org/wserv"

#' @title Path to output processed CSV
OUTPUT_FILE <- here("pipeline", "data", "processed", "usa_cesmd_pgv.csv")

#' @title Path to API response cache file
CACHE_FILE <- here("pipeline", "data", "raw", "cesmd_api_cache.rds")

#' @title Minimum magnitude threshold
#' @description Set slightly below target range (3.35-3.45) to ensure all
#'   events are captured. Final filtering happens in analysis scripts.
MAG_MIN <- 3.3

#' @title Maximum magnitude threshold
MAG_MAX <- 3.5

#' @title Maximum depth for shallow crustal events (km)
MAX_DEPTH_KM <- 15

#' @title Minimum records per event for inclusion
MIN_RECORDS <- 5

# ==============================================================================
# API Functions
# ==============================================================================

#' Query CESMD events API
#'
#' @description
#' Queries the CESMD events API for earthquakes within the specified
#' magnitude range.
#'
#' @return A tibble with event metadata:
#'   \describe{
#'     \item{eventid}{Event identifier}
#'     \item{eventdate}{Event datetime}
#'     \item{magnitude}{Event magnitude}
#'     \item{magtype}{Magnitude type (ML, Mw, etc.)}
#'     \item{latitude}{Event latitude}
#'     \item{longitude}{Event longitude}
#'     \item{depth_km}{Event depth (km)}
#'     \item{nrecords}{Number of station records available}
#'   }
#'
#' @details
#' Filters results to events with:
#' \itemize{
#'   \item At least \code{MIN_RECORDS} station records
#'   \item Depth <= \code{MAX_DEPTH_KM}
#' }
#'
#' @seealso \code{\link{query_records}}
#'
#' @export
query_events <- function() {
  message("Querying CESMD Events API...")

  url <- paste0(CESMD_BASE, "/events/query")

  resp <- request(url) %>%
    req_url_query(format = "json", minmag = MAG_MIN, maxmag = MAG_MAX) %>%
    req_timeout(60) %>%
    req_perform()

  if (resp_status(resp) != 200) stop("API error: ", resp_status(resp))

  body <- resp_body_json(resp)
  features <- body$features

  message("  Events returned: ", length(features))

  map_dfr(features, function(f) {
    props <- f$properties
    coords <- f$geometry$coordinates
    tibble(
      eventid = f$id,
      eventdate = props$time,
      magnitude = props$mag,
      magtype = props$magType,
      latitude = coords[[2]],
      longitude = coords[[1]],
      depth_km = coords[[3]],
      nrecords = props$RecordNum %||% 0
    )
  }) %>%
    filter(nrecords >= MIN_RECORDS, depth_km <= MAX_DEPTH_KM)
}

#' Query CESMD records API for a single event
#'
#' @description
#' Fetches station-level ground motion records for a specific earthquake
#' from the CESMD records API.
#'
#' @param eventid Character string identifying the earthquake event
#'
#' @return A tibble with station records:
#'   \describe{
#'     \item{eventid}{Event identifier}
#'     \item{stationcode}{Station code}
#'     \item{network}{Seismic network}
#'     \item{epidist}{Epicentral distance (km)}
#'     \item{pgv}{Peak ground velocity (cm/s), geometric mean}
#'     \item{pgav2}{Peak ground acceleration (g)}
#'     \item{vs30}{Vs30 at station (m/s)}
#'   }
#'
#' @details
#' Returns empty tibble if API request fails or no records found.
#'
#' @seealso \code{\link{query_events}}
#'
#' @export
query_records <- function(eventid) {
  url <- paste0(CESMD_BASE, "/records/query")

  resp <- request(url) %>%
    req_url_query(format = "json", rettype = "metadata",
                  eventid = eventid, groupby = "station") %>%
    req_timeout(60) %>%
    req_perform()

  if (resp_status(resp) != 200) return(tibble())

  body <- resp_body_json(resp)
  stations <- body$results$stations
  if (length(stations) == 0) return(tibble())

  map_dfr(stations, function(sta) {
    map_dfr(sta$events, function(evt) {
      rec <- evt$record
      tibble(
        eventid = eventid,
        stationcode = sta$code,
        network = sta$network,
        epidist = rec$epidist,
        pgv = rec$pgv,
        pgav2 = rec$pgav2,
        vs30 = sta$Vs30
      )
    })
  })
}

# ==============================================================================
# Processing Function
# ==============================================================================

#' Process CESMD California data to standardized format
#'
#' @description
#' Fetches California earthquake data from CESMD API and processes it to
#' standardized format compatible with the ML comparison pipeline.
#'
#' @param use_cache Logical. If TRUE (default), use cached API responses
#'   if available and less than 24 hours old.
#'
#' @return Invisibly returns a tibble with processed data.
#'   Also writes the data to \code{OUTPUT_FILE}.
#'
#' @details
#' Processing steps:
#' \enumerate{
#'   \item Check for valid cache (< 24 hours old)
#'   \item If no cache, query CESMD events API
#'   \item Fetch records for each event (with rate limiting)
#'   \item Cache raw API responses
#'   \item Process to standardized format
#' }
#'
#' Output columns (standardized schema):
#' \describe{
#'   \item{database}{Always "USA (CESMD)"}
#'   \item{event_id}{Event identifier}
#'   \item{event_date}{Event datetime as character}
#'   \item{magnitude}{Event magnitude}
#'   \item{mag_type}{Magnitude type}
#'   \item{depth_km}{Hypocentral depth (km)}
#'   \item{station_id}{Station code}
#'   \item{epicentral_dist_km}{Epicentral distance (km)}
#'   \item{pgv_mms}{PGV geometric mean (mm/s)}
#'   \item{pga_g}{PGA geometric mean (g)}
#'   \item{vs30_ms}{Vs30 at station (m/s)}
#' }
#'
#' @section Caching:
#' API responses are cached to \code{CACHE_FILE} (RDS format) for 24 hours
#' to avoid redundant API calls during development/testing.
#'
#' @examples
#' \dontrun{
#' # Use cached data if available
#' df <- process_california()
#'
#' # Force fresh API fetch
#' df <- process_california(use_cache = FALSE)
#' }
#'
#' @export
process_california <- function(use_cache = TRUE) {

  message("=" |> strrep(70))
  message("Processing CESMD (California) Strong Motion Data")
  message("=" |> strrep(70))

  # Check cache
  if (use_cache && file.exists(CACHE_FILE)) {
    cache_age <- difftime(Sys.time(), file.mtime(CACHE_FILE), units = "hours")
    if (cache_age < 24) {
      message("\nUsing cached data (", round(cache_age, 1), " hours old)")
      raw <- readRDS(CACHE_FILE)
    } else {
      raw <- NULL
    }
  } else {
    raw <- NULL
  }

  # Fetch if no cache
  if (is.null(raw)) {
    message("\nFetching fresh data from CESMD API...")

    events <- query_events()
    message("Events with >= ", MIN_RECORDS, " records: ", nrow(events))

    message("\nFetching records for each event...")
    all_records <- map_dfr(seq_len(nrow(events)), function(i) {
      if (i %% 10 == 0) message("  Progress: ", i, "/", nrow(events))
      Sys.sleep(0.3)  # Rate limiting
      query_records(events$eventid[i])
    })

    raw <- all_records %>%
      left_join(events %>% select(eventid, eventdate, magnitude, magtype, depth_km),
                by = "eventid")

    # Cache
    dir.create(dirname(CACHE_FILE), recursive = TRUE, showWarnings = FALSE)
    saveRDS(raw, CACHE_FILE)
    message("Cached to: ", CACHE_FILE)
  }

  # Process
  processed <- raw %>%
    filter(!is.na(pgv), pgv > 0) %>%
    transmute(
      database = "USA (CESMD)",
      event_id = eventid,
      event_date = eventdate,
      magnitude = magnitude,
      mag_type = toupper(magtype),
      depth_km = depth_km,
      station_id = stationcode,
      epicentral_dist_km = epidist,
      pgv_mms = pgv * 10,  # Convert cm/s to mm/s
      pga_g = pgav2,
      vs30_ms = vs30
    ) %>%
    filter(magnitude >= MAG_MIN, magnitude <= MAG_MAX)

  # Summary
  message("\n--- Processed Dataset ---")
  message("Events: ", n_distinct(processed$event_id))
  message("Records: ", nrow(processed))
  message("Magnitude types: ", paste(unique(processed$mag_type), collapse = ", "))
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
  process_california()
}
