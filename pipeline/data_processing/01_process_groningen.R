#' @title Process Groningen Ground Motion Database (GGMD)
#'
#' @description
#' Extracts and processes M3.4-3.7 earthquake ground motion data from the
#' Groningen Ground Motion Database (GGMD) Excel file.
#'
#' @details
#' Data source: KNMI/NAM Groningen Ground Motion Database
#' Download: Contact KNMI or NAM, or via Yoda repository
#' URL: https://public.yoda.uu.nl/geo/UU01/
#'
#' File required: groningenGMMdata.xlsx (place in datafiles/)
#'
#' The "GM" sheet contains geometric mean values:
#' \itemize{
#'   \item PGV_GM: Geometric mean of horizontal PGV (cm/s in source, converted to mm/s)
#'   \item PGA_GM: Geometric mean of horizontal PGA (cm/s²)
#'   \item Repi: Epicentral distance (km)
#'   \item ML: Local magnitude
#'   \item VS30: Shear wave velocity at 30m depth (m/s)
#' }
#'
#' @author Jesse
#' @date January 2025
#'
#' @name process_groningen
NULL

library(tidyverse)
library(readxl)
library(here)

# ==============================================================================
# Configuration
# ==============================================================================

#' @title Path to input GGMD Excel file
INPUT_FILE <- here("pipeline", "datafiles", "groningenGMMdata.xlsx")

#' @title Path to output processed CSV
OUTPUT_FILE <- here("pipeline", "data", "processed", "groningen_pgv.csv")

#' @title Minimum magnitude threshold
#' @description Set slightly below target range (3.35-3.45) to ensure all
#'   events are captured. Final filtering happens in analysis scripts.
MAG_MIN <- 3.3

#' @title Maximum magnitude threshold
MAG_MAX <- 3.5

# ==============================================================================
# Processing Function
# ==============================================================================

#' Process Groningen GGMD data to standardized format
#'
#' @description
#' Reads the GGMD Excel file and extracts M3.3-3.5 events with standardized
#' column names compatible with the ML comparison pipeline.
#'
#' @return Invisibly returns a tibble with processed data.
#'   Also writes the data to \code{OUTPUT_FILE}.
#'
#' @details
#' Output columns (standardized schema):
#' \describe{
#'   \item{database}{Always "Groningen"}
#'   \item{event_id}{Event identifier prefixed with "GRN_"}
#'   \item{event_date}{Event datetime as character}
#'   \item{magnitude}{Local magnitude (ML)}
#'   \item{mag_type}{Always "ML"}
#'   \item{station_id}{Station code from STAT column}
#'   \item{epicentral_dist_km}{Epicentral distance (km) from Repi column}
#'   \item{depth_km}{Hypocentral depth (km)}
#'   \item{pgv_mms}{PGV geometric mean (mm/s) from PGV_GM column}
#'   \item{pga_g}{PGA geometric mean (g) converted from PGA_GM}
#'   \item{vs30_ms}{Vs30 at station (m/s) from VS30 column}
#' }
#'
#' @section Data Quality:
#' Records are filtered to ensure:
#' \itemize{
#'   \item Valid PGV (not NA, > 0)
#'   \item Valid epicentral distance
#'   \item Magnitude within range (3.4-3.7)
#' }
#'
#' @examples
#' \dontrun{
#' # Process Groningen data
#' df <- process_groningen()
#'
#' # Check event count
#' n_distinct(df$event_id)
#' }
#'
#' @seealso \code{\link[ml_comparison_pipeline]{load_groningen_ml}}
#'
#' @export
process_groningen <- function() {

  message("=" |> strrep(70))
  message("Processing Groningen Ground Motion Database")
  message("=" |> strrep(70))

  if (!file.exists(INPUT_FILE)) {
    stop("Input file not found: ", INPUT_FILE, "\n",
         "Please download from KNMI/NAM and place in datafiles/")
  }

  message("\nReading: ", basename(INPUT_FILE))
  message("Sheet: GM")

  raw <- read_excel(INPUT_FILE, sheet = "GM")
  message("Total records: ", nrow(raw))

  # Process to standard format
  processed <- raw %>%
    transmute(
      database = "Groningen",
      event_id = paste0("GRN_", EQID),
      event_date = as.character(`Date & Time`),
      magnitude = as.numeric(ML),
      mag_type = "ML",
      station_id = as.character(STAT),
      epicentral_dist_km = as.numeric(`Repi (km)`),
      depth_km = as.numeric(`Depth (km)`),
      # PGV: Geometric mean of horizontal components (convert cm/s to mm/s)
      pgv_mms = as.numeric(PGV_GM) * 10,
      # PGA: Convert from gal (cm/s²) to g
      pga_g = as.numeric(PGA_GM) / 980.665,
      vs30_ms = as.numeric(`VS30 (m/s)`)
    ) %>%
    filter(
      !is.na(pgv_mms), pgv_mms > 0,
      !is.na(epicentral_dist_km),
      !is.na(magnitude),
      magnitude >= MAG_MIN,
      magnitude <= MAG_MAX
    )

  # Summary
  message("\n--- Processed Dataset ---")
  message("Events: ", n_distinct(processed$event_id))
  message("Records: ", nrow(processed))
  message("Magnitude range: ", round(min(processed$magnitude), 2), " - ",
          round(max(processed$magnitude), 2))
  message("Distance range: ", round(min(processed$epicentral_dist_km), 1), " - ",
          round(max(processed$epicentral_dist_km), 1), " km")
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
  process_groningen()
}
