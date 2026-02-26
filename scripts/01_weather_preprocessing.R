# ==============================================================================
# ULSAN OFFSHORE WEATHER DATA PROCESSING
# Hourly → Daily → Weekly Aggregation + Publication-Quality Figures
# Input:  data/raw/weather_hourly_raw.csv
# Output: data/processed/  (6 datasets)
#         data/outputs/     (8 statistical tables)
#         results/figures/  (10 figures)
# ==============================================================================

# 1. LIBRARY LOADING ----
library(tidyverse)
library(lubridate)
library(data.table)
library(ggplot2)
library(patchwork)
library(scales)
library(viridis)
library(gridExtra)
library(zoo)

Sys.setlocale("LC_TIME", "C")

# 2. PATHS ----
repo_root    <- here::here()   # or set manually: repo_root <- "path/to/repo"
input_file   <- file.path(repo_root, "data", "raw", "weather_hourly_raw.csv")
proc_dir     <- file.path(repo_root, "data", "processed")
out_dir      <- file.path(repo_root, "data", "outputs")
fig_dir      <- file.path(repo_root, "results", "figures")

for (d in c(proc_dir, out_dir, fig_dir)) dir.create(d, recursive = TRUE, showWarnings = FALSE)

cat("Input :", input_file, "\n")
cat("Processed output :", proc_dir, "\n")
cat("Statistical tables:", out_dir, "\n")
cat("Figures :", fig_dir, "\n\n")

cat("══════════════════════════════════════════════════════════\n")
cat("ULSAN OFFSHORE WEATHER DATA PROCESSING — HOURLY+DAILY+WEEKLY\n")
cat("══════════════════════════════════════════════════════════\n\n")

# 3. DATA LOADING ----
df_raw <- fread(input_file, encoding = "UTF-8")

df <- df_raw %>%
  rename(
    station         = `지점`,
    datetime        = `일시`,
    wind_speed      = `풍속(m/s)`,
    wind_dir        = `풍향(deg)`,
    gust_speed      = `GUST풍속(m/s)`,
    pressure        = `현지기압(hPa)`,
    humidity        = `습도(%)`,
    air_temp        = `기온(°C)`,
    sea_temp        = `수온(°C)`,
    max_wave_height = `최대파고(m)`,
    sig_wave_height = `유의파고(m)`,
    avg_wave_height = `평균파고(m)`,
    wave_period     = `파주기(sec)`,
    wave_dir        = `파향(deg)`
  ) %>%
  mutate(
    datetime    = as.POSIXct(datetime, format = "%Y-%m-%d %H:%M", tz = "Asia/Seoul"),
    date        = as.Date(datetime),
    year        = lubridate::year(datetime),
    month       = lubridate::month(datetime),
    week        = lubridate::isoweek(datetime),
    year_week   = paste0(year, "-W", sprintf("%02d", week)),
    hour        = lubridate::hour(datetime),
    day_of_week = lubridate::wday(datetime, label = TRUE)
  )

cat("Raw data loaded:", nrow(df), "hourly observations\n")
cat("Date range:", as.character(min(df$date, na.rm = TRUE)),
    "to", as.character(max(df$date, na.rm = TRUE)), "\n\n")

# 4. INTERPOLATION & QUALITY CONTROL ----
cat("Advanced interpolation with outlier removal...\n")

df <- df %>%
  mutate(across(where(is.numeric), ~ {
    med         <- median(., na.rm = TRUE)
    mad_val     <- mad(., na.rm = TRUE)
    outlier_thr <- med + 5 * mad_val
    ifelse(is.infinite(.) | abs(. - med) > outlier_thr, NA, .)
  }))

date_range  <- seq(min(df$datetime, na.rm = TRUE),
                   max(df$datetime, na.rm = TRUE), by = "hour")
complete_df <- data.frame(datetime = date_range)

df_complete <- complete_df %>%
  left_join(df, by = "datetime") %>%
  arrange(datetime) %>%
  mutate(
    date        = as.Date(datetime),
    year        = lubridate::year(datetime),
    month       = lubridate::month(datetime),
    week        = lubridate::isoweek(datetime),
    year_week   = paste0(year, "-W", sprintf("%02d", week)),
    hour        = lubridate::hour(datetime),
    day_of_week = lubridate::wday(datetime, label = TRUE)
  ) %>%
  mutate(
    wind_speed      = na.approx(wind_speed,      na.rm = FALSE, rule = 2),
    wind_dir        = na.approx(wind_dir,        na.rm = FALSE, rule = 2),
    sig_wave_height = na.approx(sig_wave_height, na.rm = FALSE, rule = 2),
    max_wave_height = na.approx(max_wave_height, na.rm = FALSE, rule = 2),
    avg_wave_height = na.approx(avg_wave_height, na.rm = FALSE, rule = 2),
    gust_speed      = na.approx(gust_speed,      na.rm = FALSE, rule = 2),
    sea_temp        = na.approx(sea_temp,        na.rm = FALSE, rule = 2),
    air_temp        = na.approx(air_temp,        na.rm = FALSE, rule = 2),
    pressure        = na.approx(pressure,        na.rm = FALSE, rule = 2),
    humidity        = na.approx(humidity,        na.rm = FALSE, rule = 2),
    wave_period     = na.approx(wave_period,     na.rm = FALSE, rule = 2),
    wave_dir        = na.approx(wave_dir,        na.rm = FALSE, rule = 2)
  ) %>%
  mutate(
    # CTV: Hs <= 1.5 m AND wind <= 10 m/s
    ctv_accessible = as.integer(sig_wave_height <= 1.5 & wind_speed <= 10),
    # SOV: Hs <= 2.5 m
    sov_accessible = as.integer(sig_wave_height <= 2.5),
    weather_condition = case_when(
      sig_wave_height > 2.5 | wind_speed > 12 ~ "Extreme",
      sig_wave_height > 1.5 | wind_speed > 10 ~ "Rough",
      TRUE                                     ~ "Calm"
    ),
    season = case_when(
      month %in% c(12, 1, 2) ~ "Winter",
      month %in% c(3, 4, 5)  ~ "Spring",
      month %in% c(6, 7, 8)  ~ "Summer",
      month %in% c(9, 10, 11)~ "Fall"
    ),
    season = factor(season, levels = c("Spring", "Summer", "Fall", "Winter")),
    time_period = case_when(
      hour >= 6  & hour < 12 ~ "Morning (06-12)",
      hour >= 12 & hour < 18 ~ "Afternoon (12-18)",
      hour >= 18 & hour < 24 ~ "Evening (18-24)",
      TRUE                   ~ "Night (00-06)"
    ),
    time_period = factor(time_period,
                         levels = c("Night (00-06)", "Morning (06-12)",
                                    "Afternoon (12-18)", "Evening (18-24)"))
  )

cat("Interpolation complete. Missing values remaining:\n")
cat("  wind_speed :", sum(is.na(df_complete$wind_speed)), "\n")
cat("  sig_wave_height:", sum(is.na(df_complete$sig_wave_height)), "\n\n")

# ==============================================================================
# 5. HOURLY DATASET ----
# ==============================================================================
cat("Preparing hourly dataset...\n")

hourly_data <- df_complete %>%
  mutate(
    wind_dir_cardinal = cut(
      wind_dir,
      breaks = c(0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360),
      labels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"),
      include.lowest = TRUE
    ),
    wind_dir_cardinal = factor(as.character(wind_dir_cardinal),
                               levels = c("N","NE","E","SE","S","SW","W","NW"))
  ) %>%
  mutate(across(where(is.numeric), ~ ifelse(is.infinite(.), NA, .)))

# Simple subset
hourly_simple <- hourly_data %>%
  select(datetime, date, year, month, hour,
         wind_speed, wind_dir,
         sig_wave_height,
         air_temp, sea_temp,
         ctv_accessible, sov_accessible,
         weather_condition, season, time_period) %>%
  rename(wave_height = sig_wave_height)

# Detailed subset
hourly_detailed <- hourly_data %>%
  select(datetime, date, year, month, week, year_week, hour, day_of_week,
         wind_speed, wind_dir, wind_dir_cardinal,
         gust_speed,
         sig_wave_height, max_wave_height, avg_wave_height,
         wave_period, wave_dir,
         air_temp, sea_temp,
         pressure, humidity,
         ctv_accessible, sov_accessible,
         weather_condition, season, time_period)

cat("Hourly dataset:", nrow(hourly_data), "rows\n\n")

# ==============================================================================
# 6. DAILY AGGREGATION ----
# ==============================================================================
cat("Daily aggregation...\n")

daily_data <- df_complete %>%
  group_by(date) %>%
  summarise(
    year        = first(year),
    month       = first(month),
    week        = first(week),
    year_week   = first(year_week),
    day_of_week = first(day_of_week),
    n_hours     = n(),

    wind_speed_mean   = mean(wind_speed, na.rm = TRUE),
    wind_speed_median = median(wind_speed, na.rm = TRUE),
    wind_speed_max    = max(wind_speed, na.rm = TRUE),
    wind_speed_min    = min(wind_speed, na.rm = TRUE),
    wind_speed_sd     = ifelse(n() > 1, sd(wind_speed, na.rm = TRUE), 0),
    wind_speed_p95    = quantile(wind_speed, 0.95, na.rm = TRUE),
    wind_speed_p05    = quantile(wind_speed, 0.05, na.rm = TRUE),

    wind_dir_mean = atan2(
      mean(sin(wind_dir * pi / 180), na.rm = TRUE),
      mean(cos(wind_dir * pi / 180), na.rm = TRUE)
    ) * 180 / pi,

    wave_height_mean   = mean(sig_wave_height, na.rm = TRUE),
    wave_height_median = median(sig_wave_height, na.rm = TRUE),
    wave_height_max    = max(sig_wave_height, na.rm = TRUE),
    wave_height_min    = min(sig_wave_height, na.rm = TRUE),
    wave_height_sd     = ifelse(n() > 1, sd(sig_wave_height, na.rm = TRUE), 0),
    wave_height_p95    = quantile(sig_wave_height, 0.95, na.rm = TRUE),
    wave_height_p05    = quantile(sig_wave_height, 0.05, na.rm = TRUE),

    max_wave_mean = mean(max_wave_height, na.rm = TRUE),
    max_wave_max  = max(max_wave_height, na.rm = TRUE),

    ctv_accessible_hours = sum(ctv_accessible, na.rm = TRUE),
    sov_accessible_hours = sum(sov_accessible, na.rm = TRUE),
    ctv_accessible_pct   = round(sum(ctv_accessible, na.rm = TRUE) / n() * 100, 1),
    sov_accessible_pct   = round(sum(sov_accessible, na.rm = TRUE) / n() * 100, 1),

    hours_calm    = sum(weather_condition == "Calm",    na.rm = TRUE),
    hours_rough   = sum(weather_condition == "Rough",   na.rm = TRUE),
    hours_extreme = sum(weather_condition == "Extreme", na.rm = TRUE),

    sea_temp_mean    = mean(sea_temp,    na.rm = TRUE),
    air_temp_mean    = mean(air_temp,    na.rm = TRUE),
    gust_speed_max   = max(gust_speed,   na.rm = TRUE),
    pressure_mean    = mean(pressure,    na.rm = TRUE),
    humidity_mean    = mean(humidity,    na.rm = TRUE),
    wave_period_mean = mean(wave_period, na.rm = TRUE),

    .groups = "drop"
  ) %>%
  mutate(
    wind_dir_mean = ifelse(wind_dir_mean < 0, wind_dir_mean + 360, wind_dir_mean),
    season = case_when(
      month %in% c(12, 1, 2) ~ "Winter",
      month %in% c(3, 4, 5)  ~ "Spring",
      month %in% c(6, 7, 8)  ~ "Summer",
      month %in% c(9, 10, 11)~ "Fall"
    ),
    season = factor(season, levels = c("Spring", "Summer", "Fall", "Winter")),
    weather_scenario = case_when(
      wind_speed_mean < 8  & wave_height_mean < 1.5    ~ "Calm",
      wind_speed_mean >= 8 & wind_speed_mean < 12 &
        wave_height_mean < 2.5                          ~ "Rough Sea",
      wind_speed_mean >= 12 | wave_height_mean >= 2.5  ~ "Extreme Weather",
      TRUE                                              ~ "Moderate"
    ),
    ctv_day_accessible = ifelse(ctv_accessible_pct >= 50, "Accessible", "Not Accessible"),
    sov_day_accessible = ifelse(sov_accessible_pct >= 50, "Accessible", "Not Accessible"),
    wind_dir_cardinal = cut(
      wind_dir_mean,
      breaks = c(0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360),
      labels = c("N","NE","E","SE","S","SW","W","NW","N"),
      include.lowest = TRUE
    ),
    wind_dir_cardinal = factor(as.character(wind_dir_cardinal),
                               levels = c("N","NE","E","SE","S","SW","W","NW"))
  ) %>%
  mutate(across(contains("_sd"), ~ ifelse(is.nan(.) | is.infinite(.), 0, .))) %>%
  mutate(across(where(is.numeric), ~ ifelse(is.infinite(.), NA, .))) %>%
  arrange(date)

cat("Daily aggregation complete:", nrow(daily_data), "days\n\n")

# ==============================================================================
# 7. WEEKLY AGGREGATION ----
# ==============================================================================
cat("Weekly aggregation...\n")

weekly_data <- df_complete %>%
  group_by(year, week, year_week) %>%
  summarise(
    week_start  = min(date, na.rm = TRUE),
    week_end    = max(date, na.rm = TRUE),
    n_hours     = n(),
    month_mode  = as.numeric(names(sort(table(month), decreasing = TRUE)[1])),

    wind_speed_mean   = mean(wind_speed, na.rm = TRUE),
    wind_speed_median = median(wind_speed, na.rm = TRUE),
    wind_speed_max    = max(wind_speed, na.rm = TRUE),
    wind_speed_min    = min(wind_speed, na.rm = TRUE),
    wind_speed_sd     = ifelse(n() > 1, sd(wind_speed, na.rm = TRUE), 0),
    wind_speed_p95    = quantile(wind_speed, 0.95, na.rm = TRUE),
    wind_speed_p05    = quantile(wind_speed, 0.05, na.rm = TRUE),

    wind_dir_mean = atan2(
      mean(sin(wind_dir * pi / 180), na.rm = TRUE),
      mean(cos(wind_dir * pi / 180), na.rm = TRUE)
    ) * 180 / pi,

    wave_height_mean   = mean(sig_wave_height, na.rm = TRUE),
    wave_height_median = median(sig_wave_height, na.rm = TRUE),
    wave_height_max    = max(sig_wave_height, na.rm = TRUE),
    wave_height_min    = min(sig_wave_height, na.rm = TRUE),
    wave_height_sd     = ifelse(n() > 1, sd(sig_wave_height, na.rm = TRUE), 0),
    wave_height_p95    = quantile(sig_wave_height, 0.95, na.rm = TRUE),
    wave_height_p05    = quantile(sig_wave_height, 0.05, na.rm = TRUE),

    max_wave_mean = mean(max_wave_height, na.rm = TRUE),
    max_wave_max  = max(max_wave_height, na.rm = TRUE),

    ctv_accessible_hours = sum(ctv_accessible, na.rm = TRUE),
    sov_accessible_hours = sum(sov_accessible, na.rm = TRUE),
    ctv_accessible_days  = sum(ctv_accessible, na.rm = TRUE) / 24,
    sov_accessible_days  = sum(sov_accessible, na.rm = TRUE) / 24,

    hours_calm    = sum(weather_condition == "Calm",    na.rm = TRUE),
    hours_rough   = sum(weather_condition == "Rough",   na.rm = TRUE),
    hours_extreme = sum(weather_condition == "Extreme", na.rm = TRUE),

    sea_temp_mean  = mean(sea_temp,  na.rm = TRUE),
    air_temp_mean  = mean(air_temp,  na.rm = TRUE),
    gust_speed_max = max(gust_speed, na.rm = TRUE),
    pressure_mean  = mean(pressure,  na.rm = TRUE),
    humidity_mean  = mean(humidity,  na.rm = TRUE),

    .groups = "drop"
  ) %>%
  mutate(
    wind_dir_mean = ifelse(wind_dir_mean < 0, wind_dir_mean + 360, wind_dir_mean),
    season = case_when(
      month_mode %in% c(12, 1, 2) ~ "Winter",
      month_mode %in% c(3, 4, 5)  ~ "Spring",
      month_mode %in% c(6, 7, 8)  ~ "Summer",
      month_mode %in% c(9, 10, 11)~ "Fall"
    ),
    season = factor(season, levels = c("Spring", "Summer", "Fall", "Winter")),
    weather_scenario = case_when(
      wind_speed_mean < 8  & wave_height_mean < 1.5    ~ "Calm",
      wind_speed_mean >= 8 & wind_speed_mean < 12 &
        wave_height_mean < 2.5                          ~ "Rough Sea",
      wind_speed_mean >= 12 | wave_height_mean >= 2.5  ~ "Extreme Weather",
      TRUE                                              ~ "Moderate"
    ),
    access_category = case_when(
      ctv_accessible_days >= 6 ~ "Excellent (>=6 days)",
      ctv_accessible_days >= 4 ~ "Good (4-6 days)",
      ctv_accessible_days >= 2 ~ "Fair (2-4 days)",
      TRUE                     ~ "Poor (<2 days)"
    ),
    access_category = factor(access_category,
                             levels = c("Excellent (>=6 days)", "Good (4-6 days)",
                                        "Fair (2-4 days)", "Poor (<2 days)")),
    wind_dir_cardinal = cut(
      wind_dir_mean,
      breaks = c(0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360),
      labels = c("N","NE","E","SE","S","SW","W","NW","N"),
      include.lowest = TRUE
    ),
    wind_dir_cardinal = factor(as.character(wind_dir_cardinal),
                               levels = c("N","NE","E","SE","S","SW","W","NW"))
  ) %>%
  arrange(year, week) %>%
  mutate(across(contains("_sd"), ~ ifelse(is.nan(.) | is.infinite(.), 0, .))) %>%
  mutate(across(where(is.numeric), ~ ifelse(is.infinite(.), NA, .)))

cat("Weekly aggregation complete:", nrow(weekly_data), "weeks\n\n")

# ==============================================================================
# 8. SAVE ALL DATASETS ----
# ==============================================================================
cat("Saving processed datasets...\n")

# Hourly
write.csv(hourly_simple,
          file.path(proc_dir, "ulsan_hourly_weather_simple.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
write.csv(hourly_detailed,
          file.path(proc_dir, "ulsan_hourly_weather_detailed.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

# Daily
daily_simple_out <- daily_data %>%
  mutate(
    wind_speed  = round(wind_speed_mean, 1),
    wind_dir    = round(wind_dir_mean, 0),
    wave_height = round(wave_height_mean, 1),
    remark      = as.character(season)
  ) %>%
  select(date, year, month, wind_speed, wind_dir, wave_height,
         ctv_accessible_hours, ctv_accessible_pct, ctv_day_accessible, remark)

write.csv(daily_simple_out,
          file.path(proc_dir, "ulsan_daily_weather_simple.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
write.csv(daily_data,
          file.path(proc_dir, "ulsan_daily_weather_detailed.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

# Weekly
weekly_simple_out <- weekly_data %>%
  mutate(
    week_of_year        = week,
    wind_speed          = round(wind_speed_mean, 1),
    wind_dir            = round(wind_dir_mean, 0),
    wave_height         = round(wave_height_mean, 1),
    turbine_access_days = round(ctv_accessible_days, 0),
    remark              = as.character(season)
  ) %>%
  select(year, week_of_year, wind_speed, wind_dir, wave_height,
         turbine_access_days, remark)

write.csv(weekly_simple_out,
          file.path(proc_dir, "ulsan_weekly_weather_simple.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
write.csv(weekly_data,
          file.path(proc_dir, "ulsan_weekly_weather_detailed.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

cat("6 datasets saved.\n\n")

# ==============================================================================
# 9. STATISTICAL TABLES ----
# ==============================================================================
cat("Computing statistical summaries...\n")

# Table 1: Weekly seasonal statistics
seasonal_stats <- weekly_data %>%
  group_by(season) %>%
  summarise(
    n_weeks          = n(),
    wind_speed_mean  = mean(wind_speed_mean,  na.rm = TRUE),
    wind_speed_sd    = ifelse(n() > 1, sd(wind_speed_mean,  na.rm = TRUE), 0),
    wave_height_mean = mean(wave_height_mean, na.rm = TRUE),
    wave_height_sd   = ifelse(n() > 1, sd(wave_height_mean, na.rm = TRUE), 0),
    ctv_access_mean  = mean(ctv_accessible_days, na.rm = TRUE),
    ctv_access_sd    = ifelse(n() > 1, sd(ctv_accessible_days, na.rm = TRUE), 0),
    scenario_calm    = sum(weather_scenario == "Calm"),
    scenario_rough   = sum(weather_scenario == "Rough Sea"),
    scenario_extreme = sum(weather_scenario == "Extreme Weather"),
    scenario_moderate= sum(weather_scenario == "Moderate"),
    pct_calm         = round(scenario_calm     / n_weeks * 100, 1),
    pct_rough        = round(scenario_rough    / n_weeks * 100, 1),
    pct_extreme      = round(scenario_extreme  / n_weeks * 100, 1),
    pct_moderate     = round(scenario_moderate / n_weeks * 100, 1),
    .groups = "drop"
  ) %>%
  mutate(across(c(wind_speed_sd, wave_height_sd, ctv_access_sd),
                ~ ifelse(is.na(.), 0, .))) %>%
  mutate(across(where(is.numeric) & !contains("pct") & !contains("n_weeks"), ~ round(., 2)))

write.csv(seasonal_stats,
          file.path(out_dir, "Table1_Seasonal_Statistics.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

# Table 1b: Daily seasonal statistics
daily_seasonal_stats <- daily_data %>%
  group_by(season) %>%
  summarise(
    n_days              = n(),
    wind_speed_mean     = round(mean(wind_speed_mean,     na.rm = TRUE), 2),
    wind_speed_sd       = round(sd(wind_speed_mean,       na.rm = TRUE), 2),
    wave_height_mean    = round(mean(wave_height_mean,    na.rm = TRUE), 2),
    wave_height_sd      = round(sd(wave_height_mean,      na.rm = TRUE), 2),
    ctv_access_pct_mean = round(mean(ctv_accessible_pct, na.rm = TRUE), 1),
    ctv_accessible_days = sum(ctv_day_accessible == "Accessible", na.rm = TRUE),
    pct_ctv_accessible  = round(ctv_accessible_days / n_days * 100, 1),
    scenario_calm       = sum(weather_scenario == "Calm"),
    scenario_rough      = sum(weather_scenario == "Rough Sea"),
    scenario_extreme    = sum(weather_scenario == "Extreme Weather"),
    scenario_moderate   = sum(weather_scenario == "Moderate"),
    pct_calm            = round(scenario_calm     / n_days * 100, 1),
    pct_rough           = round(scenario_rough    / n_days * 100, 1),
    pct_extreme         = round(scenario_extreme  / n_days * 100, 1),
    pct_moderate        = round(scenario_moderate / n_days * 100, 1),
    .groups = "drop"
  )

write.csv(daily_seasonal_stats,
          file.path(out_dir, "Table1b_Daily_Seasonal_Statistics.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

# Table 1c: Hourly seasonal statistics
hourly_seasonal_stats <- hourly_data %>%
  group_by(season) %>%
  summarise(
    n_hours              = n(),
    wind_speed_mean      = round(mean(wind_speed,      na.rm = TRUE), 2),
    wind_speed_sd        = round(sd(wind_speed,        na.rm = TRUE), 2),
    wind_speed_max       = round(max(wind_speed,       na.rm = TRUE), 2),
    wave_height_mean     = round(mean(sig_wave_height, na.rm = TRUE), 2),
    wave_height_sd       = round(sd(sig_wave_height,   na.rm = TRUE), 2),
    wave_height_max      = round(max(sig_wave_height,  na.rm = TRUE), 2),
    ctv_accessible_hours = sum(ctv_accessible, na.rm = TRUE),
    pct_ctv_accessible   = round(ctv_accessible_hours / n_hours * 100, 1),
    sov_accessible_hours = sum(sov_accessible, na.rm = TRUE),
    pct_sov_accessible   = round(sov_accessible_hours / n_hours * 100, 1),
    pct_calm             = round(sum(weather_condition == "Calm")    / n_hours * 100, 1),
    pct_rough            = round(sum(weather_condition == "Rough")   / n_hours * 100, 1),
    pct_extreme          = round(sum(weather_condition == "Extreme") / n_hours * 100, 1),
    .groups = "drop"
  )

write.csv(hourly_seasonal_stats,
          file.path(out_dir, "Table1c_Hourly_Seasonal_Statistics.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

# Table 1d: Hourly by time-of-day and season
hourly_tod_stats <- hourly_data %>%
  group_by(season, time_period) %>%
  summarise(
    n_hours          = n(),
    wind_speed_mean  = round(mean(wind_speed,      na.rm = TRUE), 2),
    wind_speed_sd    = round(sd(wind_speed,        na.rm = TRUE), 2),
    wave_height_mean = round(mean(sig_wave_height, na.rm = TRUE), 2),
    wave_height_sd   = round(sd(sig_wave_height,   na.rm = TRUE), 2),
    pct_ctv          = round(sum(ctv_accessible, na.rm = TRUE) / n_hours * 100, 1),
    pct_calm         = round(sum(weather_condition == "Calm")    / n_hours * 100, 1),
    pct_rough        = round(sum(weather_condition == "Rough")   / n_hours * 100, 1),
    pct_extreme      = round(sum(weather_condition == "Extreme") / n_hours * 100, 1),
    .groups = "drop"
  )

write.csv(hourly_tod_stats,
          file.path(out_dir, "Table1d_Hourly_TimeOfDay_Statistics.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

# Table 1e: Hourly by clock hour (0-23)
hourly_by_hour <- hourly_data %>%
  group_by(hour) %>%
  summarise(
    n_obs            = n(),
    wind_speed_mean  = round(mean(wind_speed,      na.rm = TRUE), 2),
    wind_speed_sd    = round(sd(wind_speed,        na.rm = TRUE), 2),
    wave_height_mean = round(mean(sig_wave_height, na.rm = TRUE), 2),
    wave_height_sd   = round(sd(sig_wave_height,   na.rm = TRUE), 2),
    pct_ctv          = round(sum(ctv_accessible, na.rm = TRUE) / n_obs * 100, 1),
    pct_calm         = round(sum(weather_condition == "Calm")    / n_obs * 100, 1),
    pct_rough        = round(sum(weather_condition == "Rough")   / n_obs * 100, 1),
    pct_extreme      = round(sum(weather_condition == "Extreme") / n_obs * 100, 1),
    .groups = "drop"
  )

write.csv(hourly_by_hour,
          file.path(out_dir, "Table1e_Hourly_ByHour_Statistics.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

# Table 2: Weather scenario transition matrix (weekly Markov chain)
transition_data <- weekly_data %>%
  arrange(year, week) %>%
  mutate(next_scenario = lead(weather_scenario)) %>%
  filter(!is.na(next_scenario) & !is.na(weather_scenario))

transition_matrix <- transition_data %>%
  count(weather_scenario, next_scenario) %>%
  group_by(weather_scenario) %>%
  mutate(prob = n / sum(n), prob_pct = round(prob * 100, 1)) %>%
  ungroup() %>%
  select(from_state = weather_scenario, to_state = next_scenario,
         n_transitions = n, probability_pct = prob_pct)

write.csv(transition_matrix,
          file.path(out_dir, "Table2_Weather_Transition_Matrix.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

# Table 3: Weekly CTV accessibility summary by season
access_summary <- weekly_data %>%
  group_by(season) %>%
  summarise(
    excellent_access = sum(access_category == "Excellent (>=6 days)"),
    good_access      = sum(access_category == "Good (4-6 days)"),
    fair_access      = sum(access_category == "Fair (2-4 days)"),
    poor_access      = sum(access_category == "Poor (<2 days)"),
    total_weeks      = n(),
    .groups = "drop"
  ) %>%
  mutate(
    pct_excellent = round(excellent_access / total_weeks * 100, 1),
    pct_good      = round(good_access      / total_weeks * 100, 1),
    pct_fair      = round(fair_access      / total_weeks * 100, 1),
    pct_poor      = round(poor_access      / total_weeks * 100, 1)
  )

write.csv(access_summary,
          file.path(out_dir, "Table3_Accessibility_Summary.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

# Table 3b: Daily accessibility summary by season
daily_access_summary <- daily_data %>%
  group_by(season) %>%
  summarise(
    total_days          = n(),
    ctv_accessible_days = sum(ctv_day_accessible == "Accessible", na.rm = TRUE),
    sov_accessible_days = sum(sov_day_accessible == "Accessible", na.rm = TRUE),
    pct_ctv_accessible  = round(ctv_accessible_days / total_days * 100, 1),
    pct_sov_accessible  = round(sov_accessible_days / total_days * 100, 1),
    .groups = "drop"
  )

write.csv(daily_access_summary,
          file.path(out_dir, "Table3b_Daily_Accessibility_Summary.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

cat("8 statistical tables saved.\n\n")

# ==============================================================================
# 10. VISUALIZATIONS ----
# ==============================================================================
cat("Generating publication-quality figures...\n")

theme_journal <- theme_minimal() +
  theme(
    text              = element_text(family = "sans", size = 11),
    plot.title        = element_text(face = "bold", size = 14, hjust = 0, margin = margin(b = 5)),
    plot.subtitle     = element_text(size = 10, color = "gray30", hjust = 0, margin = margin(b = 10)),
    axis.title        = element_text(size = 11, face = "bold"),
    axis.text         = element_text(size = 10, color = "gray20"),
    legend.position   = "bottom",
    legend.title      = element_text(face = "bold", size = 10),
    legend.text       = element_text(size = 9),
    panel.grid.minor  = element_blank(),
    panel.grid.major  = element_line(color = "gray90", linewidth = 0.3),
    plot.margin       = margin(15, 15, 15, 15),
    strip.text        = element_text(face = "bold", size = 10),
    strip.background  = element_rect(fill = "gray95", color = NA)
  )

# Figure 1: Weekly time series
p1a <- ggplot(weekly_data, aes(x = week_start)) +
  geom_line(aes(y = wind_speed_mean, color = "Wind Speed"), linewidth = 0.9, alpha = 0.9) +
  geom_line(aes(y = wave_height_mean * 5, color = "Wave Height (x5)"), linewidth = 0.9, alpha = 0.9) +
  geom_ribbon(aes(ymin = pmax(wind_speed_mean - wind_speed_sd, 0),
                  ymax = wind_speed_mean + wind_speed_sd),
              alpha = 0.2, fill = "#E31A1C") +
  geom_hline(yintercept = 10,      linetype = "dashed", color = "#0066CC", alpha = 0.6, linewidth = 0.6) +
  geom_hline(yintercept = 1.5 * 5, linetype = "dashed", color = "#FF8C00", alpha = 0.6, linewidth = 0.6) +
  annotate("text", x = min(weekly_data$week_start) + 60, y = 10.5,
           label = "CTV Wind Limit (10 m/s)", size = 3, color = "#0066CC", hjust = 0) +
  annotate("text", x = min(weekly_data$week_start) + 60, y = 8.0,
           label = "CTV Wave Limit (1.5 m)",  size = 3, color = "#FF8C00", hjust = 0) +
  scale_y_continuous(name = "Wind Speed (m/s)",
                     sec.axis = sec_axis(~ . / 5, name = "Significant Wave Height (m)")) +
  scale_color_manual(values = c("Wind Speed" = "#E31A1C", "Wave Height (x5)" = "#1F78B4"), name = "") +
  labs(title = "A. Weekly Wind Speed and Wave Height Trends",
       subtitle = "Ulsan offshore marine observations (2023-2025) with ±1 SD bands",
       x = "Date") +
  theme_journal + theme(legend.position = c(0.15, 0.95))

p1b <- ggplot(weekly_data, aes(x = week_start, y = ctv_accessible_days, fill = season)) +
  geom_col(alpha = 0.85, width = 5) +
  geom_hline(yintercept = 4, linetype = "dashed", color = "#CC0000", linewidth = 0.7) +
  annotate("text", x = min(weekly_data$week_start) + 45, y = 4.4,
           label = "Target: 4 days/week", color = "#CC0000", size = 3.5, hjust = 0, fontface = "bold") +
  scale_fill_viridis_d(option = "D", name = "Season", begin = 0.15, end = 0.85) +
  scale_y_continuous(breaks = seq(0, 7, 1), limits = c(0, 7.8)) +
  labs(title = "B. CTV Operational Accessibility per Week",
       subtitle = "Access criteria: Wave height ≤1.5 m AND wind speed ≤10 m/s",
       x = "Date", y = "Accessible Days (days/week)") +
  theme_journal

ggsave(file.path(fig_dir, "Fig1_Weather_Trends_Enhanced.png"),
       (p1a / p1b) + plot_layout(heights = c(1.2, 1)),
       width = 13, height = 11, dpi = 300, bg = "white")

# Figure 2: Seasonal box plots
p2a <- ggplot(weekly_data, aes(x = season, y = wind_speed_mean, fill = season)) +
  geom_boxplot(alpha = 0.75, outlier.shape = 21, outlier.size = 2.5, width = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.35, size = 1.8, color = "gray30") +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 4,
               fill = "white", color = "black", stroke = 1.2) +
  scale_fill_viridis_d(option = "C", guide = "none", begin = 0.2, end = 0.9) +
  labs(title = "C. Seasonal Wind Speed Distribution",
       subtitle = "Diamond markers indicate seasonal mean values",
       x = "Season", y = "Mean Weekly Wind Speed (m/s)") +
  theme_journal

p2b <- ggplot(weekly_data, aes(x = season, y = wave_height_mean, fill = season)) +
  geom_boxplot(alpha = 0.75, outlier.shape = 21, outlier.size = 2.5, width = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.35, size = 1.8, color = "gray30") +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 4,
               fill = "white", color = "black", stroke = 1.2) +
  scale_fill_viridis_d(option = "C", guide = "none", begin = 0.2, end = 0.9) +
  labs(title = "D. Seasonal Wave Height Distribution",
       subtitle = "Diamond markers indicate seasonal mean values",
       x = "Season", y = "Mean Significant Wave Height (m)") +
  theme_journal

ggsave(file.path(fig_dir, "Fig2_Seasonal_Distributions_Enhanced.png"),
       (p2a | p2b), width = 13, height = 6, dpi = 300, bg = "white")

# Figure 3: Multi-panel analysis
p3a <- ggplot(weekly_data, aes(x = wind_speed_mean, y = wave_height_mean)) +
  geom_point(aes(color = weather_scenario, size = ctv_accessible_days), alpha = 0.7) +
  geom_hline(yintercept = 1.5, linetype = "dashed", color = "#0066CC", linewidth = 0.7) +
  geom_hline(yintercept = 2.5, linetype = "dashed", color = "#FF6600", linewidth = 0.7) +
  geom_vline(xintercept = 10,  linetype = "dashed", color = "#0066CC", linewidth = 0.7) +
  geom_vline(xintercept = 12,  linetype = "dashed", color = "#FF6600", linewidth = 0.7) +
  annotate("text", x = 10.5, y = 1.35, label = "CTV Threshold", size = 3.2, color = "#0066CC", hjust = 0, fontface = "bold") +
  annotate("text", x = 12.5, y = 2.35, label = "SOV Threshold", size = 3.2, color = "#FF6600", hjust = 0, fontface = "bold") +
  scale_color_manual(
    values = c("Calm" = "#2E7D32", "Rough Sea" = "#F57C00",
               "Extreme Weather" = "#C62828", "Moderate" = "#1976D2"),
    name = "Weather\nScenario") +
  scale_size_continuous(range = c(2, 9), name = "CTV Access\nDays") +
  labs(title = "E. Wind Speed vs. Wave Height",
       subtitle = "Operational thresholds for CTV and SOV vessels",
       x = "Mean Wind Speed (m/s)", y = "Mean Significant Wave Height (m)") +
  theme_journal +
  guides(color = guide_legend(override.aes = list(size = 5), order = 1),
         size  = guide_legend(order = 2))

wind_rose_data <- weekly_data %>%
  filter(!is.na(wind_dir_cardinal)) %>%
  mutate(wind_speed_bin = cut(wind_speed_mean,
                              breaks = c(0, 5, 10, 15, Inf),
                              labels = c("<5","5-10","10-15",">15"))) %>%
  filter(!is.na(wind_speed_bin)) %>%
  count(wind_dir_cardinal, wind_speed_bin)

p3b <- ggplot(wind_rose_data, aes(x = wind_dir_cardinal, y = n, fill = wind_speed_bin)) +
  geom_col(position = "stack", alpha = 0.85, color = "white", linewidth = 0.4) +
  coord_polar(start = -pi / 8) +
  scale_fill_viridis_d(option = "B", name = "Wind Speed\n(m/s)", begin = 0.2, end = 0.95) +
  labs(title = "F. Wind Rose Diagram",
       subtitle = "Prevailing wind direction and speed", x = "", y = "Frequency (weeks)") +
  theme_journal +
  theme(axis.text.y = element_text(size = 8),
        panel.grid.major = element_line(color = "gray80", linewidth = 0.3))

scenario_summary <- weekly_data %>%
  count(weather_scenario) %>%
  arrange(desc(n)) %>%
  mutate(pct = n / sum(n) * 100,
         label_text = paste0(round(pct, 1), "%\n(n=", n, ")"))

p3c <- ggplot(scenario_summary,
              aes(x = reorder(weather_scenario, -n), y = n, fill = weather_scenario)) +
  geom_col(alpha = 0.85, color = "white", linewidth = 1.5, width = 0.7) +
  geom_text(aes(label = label_text), vjust = -0.3, size = 4.5,
            fontface = "bold", color = "black", lineheight = 0.85) +
  scale_fill_manual(
    values = c("Calm" = "#A8E6CF", "Rough Sea" = "#FFD3B6",
               "Extreme Weather" = "#FFAAA5", "Moderate" = "#A7C7E7"),
    name = "Weather Scenario") +
  scale_y_continuous(limits = c(0, max(scenario_summary$n) * 1.15),
                     breaks = seq(0, max(scenario_summary$n), by = 20),
                     expand = c(0, 0)) +
  labs(title = "G. Weather Scenario Distribution (2023-2025)",
       subtitle = paste0("Frequency of weather conditions over ", nrow(weekly_data), " weeks"),
       x = "", y = "Number of Weeks") +
  theme_journal +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 0, hjust = 0.5, face = "bold", size = 10),
        panel.grid.major.x = element_blank())

heatmap_data <- weekly_data %>%
  select(year, week, ctv_accessible_days) %>%
  complete(year = unique(year), week = 1:52)

p3d <- ggplot(heatmap_data, aes(x = week, y = as.factor(year), fill = ctv_accessible_days)) +
  geom_tile(color = "white", linewidth = 0.6) +
  scale_fill_viridis_c(option = "A", name = "Access Days\n(days/week)",
                       na.value = "gray92", limits = c(0, 7), breaks = 0:7) +
  scale_x_continuous(breaks = seq(1, 52, by = 4), expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  labs(title = "H. Weekly CTV Accessibility Heatmap",
       subtitle = "Darker = lower accessibility (winter pattern visible)",
       x = "Week of Year", y = "Year") +
  theme_journal + theme(panel.grid = element_blank())

ggsave(file.path(fig_dir, "Fig3_Comprehensive_Analysis_Fixed.png"),
       (p3a | p3b) / (p3c | p3d) + plot_layout(heights = c(1, 1.3)),
       width = 15, height = 12, dpi = 300, bg = "white")

# Figure 4: Transition matrix heatmap
p4 <- ggplot(transition_matrix, aes(x = from_state, y = to_state, fill = probability_pct)) +
  geom_tile(color = "white", linewidth = 2) +
  geom_text(aes(label = paste0(probability_pct, "%")),
            size = 5.5, fontface = "bold", color = "white") +
  scale_fill_viridis_c(option = "C", name = "Transition\nProbability (%)",
                       limits = c(0, 100), breaks = seq(0, 100, 25)) +
  labs(title = "Weather Scenario Transition Probability Matrix",
       subtitle = "For Dynamic Bayesian Network (DBN) modelling — Markov chain transitions",
       x = "Current Week Scenario", y = "Next Week Scenario") +
  theme_journal +
  theme(panel.grid = element_blank(),
        axis.text.x = element_text(angle = 0, hjust = 0.5, face = "bold"),
        axis.text.y = element_text(face = "bold"))

ggsave(file.path(fig_dir, "Fig4_Transition_Matrix_Enhanced.png"),
       p4, width = 11, height = 8, dpi = 300, bg = "white")

# Figure 5: Monthly wind patterns by year
monthly_data <- df_complete %>%
  mutate(month_name = factor(month.name[month], levels = month.name)) %>%
  group_by(year, month, month_name) %>%
  summarise(wind_speed_mean = mean(wind_speed, na.rm = TRUE), .groups = "drop")

p5 <- ggplot(monthly_data, aes(x = factor(year), y = wind_speed_mean, fill = month_name)) +
  geom_col(position = "dodge", alpha = 0.85, width = 0.7) +
  facet_wrap(~ month_name, nrow = 3, scales = "free_y") +
  scale_fill_viridis_d(option = "D", guide = "none", begin = 0.1, end = 0.95) +
  labs(title = "Monthly Wind Speed Patterns Across Years",
       subtitle = "Comparison of mean wind speeds by month (2023-2025)",
       x = "Year", y = "Mean Wind Speed (m/s)") +
  theme_journal +
  theme(strip.background = element_rect(fill = "gray92", color = "gray70", linewidth = 0.5),
        panel.spacing = unit(1, "lines"))

ggsave(file.path(fig_dir, "Fig5_Monthly_Patterns_Enhanced.png"),
       p5, width = 13, height = 9, dpi = 300, bg = "white")

# Figure 6: Daily CTV accessibility heatmap
daily_heatmap_data <- daily_data %>%
  mutate(day_of_year = lubridate::yday(date))

p6 <- ggplot(daily_heatmap_data,
             aes(x = day_of_year, y = as.factor(year), fill = ctv_accessible_pct)) +
  geom_tile(color = "white", linewidth = 0.3) +
  scale_fill_viridis_c(option = "A", name = "CTV Accessible\nHours (%)",
                       na.value = "gray92", limits = c(0, 100), breaks = seq(0, 100, 25)) +
  scale_x_continuous(
    breaks = c(1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335),
    labels = c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"),
    expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  labs(title = "Daily CTV Accessibility Heatmap (2023-2025)",
       subtitle = "% of hours per day meeting CTV criteria (wave ≤1.5 m AND wind ≤10 m/s)",
       x = "Month", y = "Year") +
  theme_journal +
  theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 0, hjust = 0.5))

ggsave(file.path(fig_dir, "Fig6_Daily_CTV_Accessibility_Heatmap.png"),
       p6, width = 14, height = 6, dpi = 300, bg = "white")

# Figure 7: Hourly wind/wave by hour-of-day and season
p7a <- ggplot(hourly_data, aes(x = factor(hour), y = wind_speed, fill = season)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.5, outlier.alpha = 0.3, width = 0.7) +
  scale_fill_viridis_d(option = "D", name = "Season", begin = 0.15, end = 0.85) +
  labs(title = "I. Hourly Wind Speed by Time-of-Day and Season",
       subtitle = "Distribution of wind speed across 24 hours",
       x = "Hour of Day (0-23)", y = "Wind Speed (m/s)") +
  theme_journal + theme(axis.text.x = element_text(size = 8))

p7b <- ggplot(hourly_data, aes(x = factor(hour), y = sig_wave_height, fill = season)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.5, outlier.alpha = 0.3, width = 0.7) +
  scale_fill_viridis_d(option = "D", name = "Season", begin = 0.15, end = 0.85) +
  labs(title = "J. Hourly Wave Height by Time-of-Day and Season",
       subtitle = "Distribution of significant wave height across 24 hours",
       x = "Hour of Day (0-23)", y = "Significant Wave Height (m)") +
  theme_journal + theme(axis.text.x = element_text(size = 8))

ggsave(file.path(fig_dir, "Fig7_Hourly_WindWave_ByHour.png"),
       (p7a / p7b), width = 14, height = 10, dpi = 300, bg = "white")

# Figure 8: Hourly CTV accessibility by hour and season
ctv_by_hour_season <- hourly_data %>%
  group_by(hour, season) %>%
  summarise(pct_ctv = round(mean(ctv_accessible, na.rm = TRUE) * 100, 1), .groups = "drop")

p8 <- ggplot(ctv_by_hour_season, aes(x = hour, y = pct_ctv, color = season, group = season)) +
  geom_line(linewidth = 1.2, alpha = 0.9) +
  geom_point(size = 2.5, alpha = 0.9) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "gray40", linewidth = 0.6) +
  annotate("text", x = 0.5, y = 51.5, label = "50% threshold", size = 3, color = "gray40", hjust = 0) +
  scale_color_viridis_d(option = "D", name = "Season", begin = 0.15, end = 0.85) +
  scale_x_continuous(breaks = 0:23) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 20)) +
  labs(title = "K. CTV Accessibility Rate by Hour-of-Day and Season",
       subtitle = "% of observations meeting CTV criteria (wave ≤1.5 m AND wind ≤10 m/s)",
       x = "Hour of Day (0-23)", y = "CTV Accessible (%)") +
  theme_journal

ggsave(file.path(fig_dir, "Fig8_Hourly_CTV_ByHour_Season.png"),
       p8, width = 13, height = 6, dpi = 300, bg = "white")

# Figure 9: Hourly condition heatmap (hour × month)
hourly_heatmap <- hourly_data %>%
  group_by(month, hour) %>%
  summarise(
    pct_ctv     = round(mean(ctv_accessible,                na.rm = TRUE) * 100, 1),
    pct_calm    = round(mean(weather_condition == "Calm",   na.rm = TRUE) * 100, 1),
    pct_extreme = round(mean(weather_condition == "Extreme",na.rm = TRUE) * 100, 1),
    .groups = "drop"
  ) %>%
  mutate(month_name = factor(month.abb[month], levels = month.abb))

p9a <- ggplot(hourly_heatmap, aes(x = hour, y = month_name, fill = pct_ctv)) +
  geom_tile(color = "white", linewidth = 0.4) +
  scale_fill_viridis_c(option = "A", name = "CTV\nAccessible (%)",
                       limits = c(0, 100), breaks = seq(0, 100, 25)) +
  scale_x_continuous(breaks = seq(0, 23, 3), expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  labs(title = "L. CTV Accessibility Rate: Hour × Month Heatmap",
       subtitle = "% of hours meeting CTV criteria per (month, hour) cell",
       x = "Hour of Day", y = "Month") +
  theme_journal + theme(panel.grid = element_blank())

p9b <- ggplot(hourly_heatmap, aes(x = hour, y = month_name, fill = pct_extreme)) +
  geom_tile(color = "white", linewidth = 0.4) +
  scale_fill_viridis_c(option = "C", name = "Extreme\nCondition (%)",
                       limits = c(0, max(hourly_heatmap$pct_extreme, na.rm = TRUE))) +
  scale_x_continuous(breaks = seq(0, 23, 3), expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  labs(title = "M. Extreme Weather Rate: Hour × Month Heatmap",
       subtitle = "% of hours with sig. wave >2.5 m OR wind >12 m/s per (month, hour) cell",
       x = "Hour of Day", y = "Month") +
  theme_journal + theme(panel.grid = element_blank())

ggsave(file.path(fig_dir, "Fig9_Hourly_Heatmap_Month_x_Hour.png"),
       (p9a / p9b), width = 13, height = 10, dpi = 300, bg = "white")

# Figure 10: Time-of-day seasonal summary
tod_season_summary <- hourly_data %>%
  group_by(season, time_period) %>%
  summarise(
    wind_mean = mean(wind_speed,      na.rm = TRUE),
    wind_sd   = sd(wind_speed,        na.rm = TRUE),
    wave_mean = mean(sig_wave_height, na.rm = TRUE),
    wave_sd   = sd(sig_wave_height,   na.rm = TRUE),
    pct_ctv   = mean(ctv_accessible,  na.rm = TRUE) * 100,
    .groups = "drop"
  )

p10a <- ggplot(tod_season_summary, aes(x = time_period, y = wind_mean, fill = season)) +
  geom_col(position = "dodge", alpha = 0.85, width = 0.75) +
  geom_errorbar(aes(ymin = pmax(wind_mean - wind_sd, 0), ymax = wind_mean + wind_sd),
                position = position_dodge(0.75), width = 0.25, linewidth = 0.6) +
  scale_fill_viridis_d(option = "D", name = "Season", begin = 0.15, end = 0.85) +
  labs(title = "N. Mean Wind Speed by Time-of-Day and Season",
       subtitle = "Error bars = ±1 SD",
       x = "Time Period", y = "Mean Wind Speed (m/s)") +
  theme_journal + theme(axis.text.x = element_text(angle = 15, hjust = 1))

p10b <- ggplot(tod_season_summary, aes(x = time_period, y = pct_ctv, fill = season)) +
  geom_col(position = "dodge", alpha = 0.85, width = 0.75) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "gray40", linewidth = 0.6) +
  scale_fill_viridis_d(option = "D", name = "Season", begin = 0.15, end = 0.85) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 20)) +
  labs(title = "O. CTV Accessibility Rate by Time-of-Day and Season",
       subtitle = "Dashed line = 50% accessibility threshold",
       x = "Time Period", y = "CTV Accessible (%)") +
  theme_journal + theme(axis.text.x = element_text(angle = 15, hjust = 1))

ggsave(file.path(fig_dir, "Fig10_Hourly_TimeOfDay_Season.png"),
       (p10a | p10b), width = 14, height = 6, dpi = 300, bg = "white")

cat("10 figures saved.\n\n")

# ==============================================================================
# 11. FINAL SUMMARY ----
# ==============================================================================
cat("\n══════════════════════════════════════════════════════════\n")
cat("PROCESSING COMPLETE — HOURLY + DAILY + WEEKLY\n")
cat("══════════════════════════════════════════════════════════\n\n")

cat("Processed datasets:", proc_dir, "\n")
cat("  ulsan_hourly_weather_simple.csv\n")
cat("  ulsan_hourly_weather_detailed.csv\n")
cat("  ulsan_daily_weather_simple.csv\n")
cat("  ulsan_daily_weather_detailed.csv\n")
cat("  ulsan_weekly_weather_simple.csv\n")
cat("  ulsan_weekly_weather_detailed.csv\n\n")

cat("Statistical tables:", out_dir, "\n")
cat("  Table1_Seasonal_Statistics.csv\n")
cat("  Table1b_Daily_Seasonal_Statistics.csv\n")
cat("  Table1c_Hourly_Seasonal_Statistics.csv\n")
cat("  Table1d_Hourly_TimeOfDay_Statistics.csv\n")
cat("  Table1e_Hourly_ByHour_Statistics.csv\n")
cat("  Table2_Weather_Transition_Matrix.csv\n")
cat("  Table3_Accessibility_Summary.csv\n")
cat("  Table3b_Daily_Accessibility_Summary.csv\n\n")

cat("Figures:", fig_dir, "\n")
cat("  Fig1_Weather_Trends_Enhanced.png\n")
cat("  Fig2_Seasonal_Distributions_Enhanced.png\n")
cat("  Fig3_Comprehensive_Analysis_Fixed.png\n")
cat("  Fig4_Transition_Matrix_Enhanced.png\n")
cat("  Fig5_Monthly_Patterns_Enhanced.png\n")
cat("  Fig6_Daily_CTV_Accessibility_Heatmap.png\n")
cat("  Fig7_Hourly_WindWave_ByHour.png\n")
cat("  Fig8_Hourly_CTV_ByHour_Season.png\n")
cat("  Fig9_Hourly_Heatmap_Month_x_Hour.png\n")
cat("  Fig10_Hourly_TimeOfDay_Season.png\n\n")

cat("Hourly Seasonal Statistics:\n")
print(hourly_seasonal_stats)
