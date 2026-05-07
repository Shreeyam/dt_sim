"""Physical and time constants."""
import datetime


class Constants:
    J2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)
    seconds_per_day = 24 * 60 * 60

    R_E = 6378.0           # Earth radius, km
    mu = 398600.4418       # Earth gravitational parameter, km^3/s^2
    ERA_J2000 = 280.46     # Earth rotation angle at J2000, deg
    gamma = 360.9856123035484  # Earth rotation rate, deg/day
