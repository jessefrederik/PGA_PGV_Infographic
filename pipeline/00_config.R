#' @title ML-only PGV Comparison Pipeline Configuration
#'
#' @description
#' Configuration file defining standardized settings for comparing M3.4
#' earthquake ground motions across countries using only Local Magnitude (ML).
#'
#' @details
#' This configuration ensures consistent data filtering and processing across

#' all comparison scripts. Key settings include:
#' \itemize{
#'   \item Magnitude filter: M3.35-3.45 (centered on M3.4)
#'   \item Distance filter: <= 50 km (Groningen network limit)
#'   \item PGV definition: Geometric mean of horizontal components
#'   \item Colorblind-friendly palette (Okabe-Ito)
#' }
#'
#' @author Jesse
#' @date January 2025
#'
#' @name config
NULL

library(tidyverse)
library(here)

# ==============================================================================
# MAGNITUDE FILTER
# ==============================================================================

#' @title Minimum magnitude threshold
#' @description Lower bound for magnitude filter, centered on M3.4
#' @export
MAG_MIN <- 3.35

#' @title Maximum magnitude threshold
#' @description Upper bound for magnitude filter, centered on M3.4
#' @export
MAG_MAX <- 3.45

# ==============================================================================
# DISTANCE FILTER
# ==============================================================================

#' @title Maximum epicentral distance for analysis
#' @description Limited to Groningen network maximum for fair comparison
#' @export
DIST_MAX_KM <- 50

# ==============================================================================
# DEPTH FILTER
# ==============================================================================

#' @title Maximum hypocentral depth for analysis
#' @description Shallow crustal earthquakes only (< 20 km)
#' @export
DEPTH_MAX_KM <- 20

# ==============================================================================
# PGV DEFINITION REFERENCE
# ==============================================================================
#
# CRITICAL: All databases must use geometric mean of horizontal components
#
# Database        | PGV Column      | Definition
# ----------------|-----------------|------------------------------------------
# Groningen GGMD  | PGV_GM          | Geometric mean (provided)
# Italy ITACA     | sqrt(E x N)     | Computed from E/N components
# Turkey SMD-TR   | pgaGeo_(cm/s)   | Geometric mean (from IM_Geo.csv)
# Japan J-SHIS    | sqrt(H1 x H2)   | Computed from maxvel1/maxvel2
# USA CESMD       | pgv             | Geometric mean (API returns GM)
#
# Note: USA NGA-West2 uses RotD50 (Mw only) - excluded from ML comparison

# ==============================================================================
# UNIT CONVERSION FUNCTIONS
# ==============================================================================

#' Convert acceleration from gal to g
#'
#' @description Converts acceleration values from gal (cm/s²) to g units.
#'
#' @param x Numeric vector of acceleration values in gal (cm/s²)
#'
#' @return Numeric vector of acceleration values in g
#'
#' @details
#' Standard gravity g = 980.665 cm/s² (gal)
#'
#' @examples
#' gal_to_g(980.665)  # Returns 1.0
#' gal_to_g(c(100, 500, 1000))
#'
#' @export
gal_to_g <- function(x) x / 980.665

#' Convert acceleration from m/s² to g
#'
#' @description Converts acceleration values from m/s² to g units.
#'
#' @param x Numeric vector of acceleration values in m/s²
#'
#' @return Numeric vector of acceleration values in g
#'
#' @details
#' Standard gravity g = 9.80665 m/s²
#'
#' @examples
#' ms2_to_g(9.80665)  # Returns 1.0
#' ms2_to_g(c(1, 5, 10))
#'
#' @export
ms2_to_g <- function(x) x / 9.80665

#' Compute geometric mean of two values
#'
#' @description Calculates the geometric mean of two numeric values,
#' typically used for combining horizontal ground motion components.
#'
#' @param x First numeric value (e.g., NS component)
#' @param y Second numeric value (e.g., EW component)
#'
#' @return Numeric geometric mean: sqrt(x * y)
#'
#' @details
#' The geometric mean is the standard method for combining horizontal
#' ground motion components in seismology. It is less sensitive to
#' outliers than the arithmetic mean and represents the "average"
#' horizontal motion.
#'
#' @examples
#' geom_mean(2, 8)    # Returns 4
#' geom_mean(0.5, 2)  # Returns 1
#'
#' @export
geom_mean <- function(x, y) sqrt(x * y)

# ==============================================================================
# STANDARDIZED OUTPUT SCHEMA
# ==============================================================================

#' @title Standard column names for processed data
#' @description
#' All processing scripts produce data frames with these standardized columns
#' to ensure consistency across databases.
#'
#' @details
#' Column definitions:
#' \describe{
#'   \item{country}{Country name (character)}
#'   \item{event_id}{Unique event identifier (character)}
#'   \item{magnitude}{Local magnitude ML or JMA (numeric)}
#'   \item{mag_type}{Magnitude type: "ML" or "JMA" (character)}
#'   \item{depth_km}{Hypocentral depth in km (numeric, may be NA)}
#'   \item{epicentral_dist_km}{Epicentral distance in km (numeric)}
#'   \item{pgv_cms}{PGV geometric mean in cm/s (numeric)}
#' }
#'
#' @export
STANDARD_COLS <- c(
  "country",
  "event_id",
  "magnitude",
  "mag_type",
  "depth_km",
  "epicentral_dist_km",
  "pgv_cms"
)

# ==============================================================================
# COLOR PALETTE
# ==============================================================================

#' @title Country color palette
#' @description
#' Okabe-Ito colorblind-friendly palette for consistent visualization
#' across all comparison plots.
#'
#' @details
#' Colors are optimized for distinguishability by colorblind viewers.
#' Netherlands (orange) is highlighted as the reference category.
#'
#' @export
COUNTRY_COLORS <- c(
  "Netherlands" = "#E69F00",
  "Italy" = "#009E73",
  "California" = "#56B4E9",
  "Japan" = "#D55E00",
  "Turkey" = "#CC79A7"
)

# ==============================================================================
# DATA FILE PATHS
# ==============================================================================

#' @title Paths to input data files
#' @description
#' Named list of file paths for each database source.
#' Uses \code{here::here()} for project-relative paths.
#'
#' @details
#' Data sources:
#' \describe{
#'   \item{groningen}{GGMD Excel file with ground motion data}
#'   \item{italy}{Processed ITACA 3.2 CSV}
#'   \item{turkey}{SMD-TR database directory}
#'   \item{california}{Processed CESMD API data CSV}
#'   \item{japan}{Processed J-SHIS flatfile CSV}
#' }
#'
#' @export
DATA_PATHS <- list(
  groningen  = here("pipeline", "datafiles", "groningenGMMdata.xlsx"),
  italy      = here("pipeline", "data", "processed", "italy_itaca_pgv.csv"),
  turkey     = here("pipeline", "datafiles", "PRJ-3950",
                    "Project--an-updated-strong-motion-database-of-turkiye-smd-tr--V3",
                    "data"),
  california = here("pipeline", "data", "processed", "usa_cesmd_pgv.csv"),
  japan      = here("pipeline", "data", "processed", "japan_jshis_pgv.csv")
)

# ==============================================================================
# OUTPUT PATHS
# ==============================================================================

#' @title Output directory for figures and tables
#' @description
#' Directory where all pipeline outputs (PNG, PDF, CSV) are saved.
#' Created automatically if it doesn't exist.
#'
#' @export
OUTPUT_DIR <- here("pipeline", "output")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
