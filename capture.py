#!/usr/bin/env python3
"""
MicaSense RedEdge capture script.

1. Checks camera status/connectivity.
2. Performs a panel calibration capture (radiometric calibration via the
   MicaSense reflectance panel, using the /detect_panel capture option).
3. Continuously captures images at a fixed interval for a set duration.

API reference: https://micasense.github.io/rededge-api/api/http.html
"""

import argparse
import sys
import time
from datetime import datetime

import requests

DEFAULT_IP = "192.168.10.254"  # default Ethernet IP 192.168.1.83; use 192.168.10.254 over WiFi AP


def get_status(base_url, timeout=5):
    resp = requests.get(f"{base_url}/status", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def print_status_summary(status):
    print("Camera status:")
    print(f"  SD card: {status.get('sd_status')} "
          f"({status.get('sd_gb_free'):.2f} / {status.get('sd_gb_total'):.2f} GB free)")
    print(f"  Bus volts: {status.get('bus_volts')}")
    print(f"  GPS: {status.get('gps_type')} "
          f"({status.get('gps_used_sats')} sats used)")


def calibrate_with_panel(base_url, timeout=30, max_attempts=1):
    """
    Capture an image of the MicaSense calibrated reflectance panel.
    Point the camera straight down at the panel in good, even lighting
    before calling this. 

    NOTE ON PLACEMENT: Need to put the camera right over the panel, about a meter above it. Too much higher and it wont detect it. 
    If its on the robot use a table. Or detach it. I'm not your dad do what you want.
    """
    print("\n--- Panel calibration capture ---")
    print("Point the camera at the reflectance panel now.")
    input("Press Enter when ready...")

    params = {
        "detect_panel": "true",   # camera waits until it detects the panel
        "block": "true",          # don't return until capture completes
        "store_capture": "true",
    }

    for attempt in range(1, max_attempts + 1):
        print(f"Capturing panel image (attempt {attempt}/{max_attempts})...")
        resp = requests.get(f"{base_url}/capture", params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "complete":
            print(f"Panel calibration capture complete. Capture ID: {data.get('id')}")
            return data
        else:
            print(f"Panel capture did not complete cleanly: {data}")

    raise RuntimeError("Panel calibration capture failed after all attempts.")


def capture_once(base_url, timeout=10):
    params = {"store_capture": "true", "block": "true"}
    resp = requests.get(f"{base_url}/capture", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def run_continuous_capture(base_url, duration_s, interval_s):
    print(f"\n--- Continuous capture: every {interval_s}s for {duration_s}s ---")
    end_time = time.time() + duration_s
    count = 0
    errors = 0

    while time.time() < end_time:
        loop_start = time.time()
        try:
            data = capture_once(base_url)
            count += 1
            ts = data.get("time", datetime.utcnow().isoformat())
            print(f"[{count}] id={data.get('id')} status={data.get('status')} time={ts}")
        except requests.exceptions.RequestException as e:
            errors += 1
            print(f"Capture failed: {e}")

        elapsed = time.time() - loop_start
        remaining = interval_s - elapsed
        if remaining > 0:
            time.sleep(remaining)
        else:
            print(f"  Warning: capture took {elapsed:.2f}s, longer than "
                  f"the {interval_s}s interval.")

    print(f"\nDone. {count} captures attempted, {errors} failed.")


def main():
    parser = argparse.ArgumentParser(description="MicaSense RedEdge capture script")
    parser.add_argument("--ip", default=DEFAULT_IP,
                         help=f"Camera IP address (default: {DEFAULT_IP})")
    parser.add_argument("--duration", type=float, default=300,
                         help="Total capture duration in seconds (default: 300 = 5 min)")
    parser.add_argument("--interval", type=float, default=1.0,
                         help="Seconds between captures (default: 1.0)")
    parser.add_argument("--skip-calibration", action="store_true",
                         help="Skip the panel calibration capture step")
    args = parser.parse_args()

    base_url = f"http://{args.ip}"

    try:
        status = get_status(base_url)
    except requests.exceptions.RequestException as e:
        print(f"Could not reach camera at {base_url}: {e}")
        sys.exit(1)

    print_status_summary(status)

    if status.get("sd_status") != "Ok":
        print(f"Warning: SD card status is '{status.get('sd_status')}', not 'Ok'.")

    if not args.skip_calibration:
        calibrate_with_panel(base_url)
        print("\nMove the camera into position for data collection.")
        input("Press Enter to start continuous capture...")
    else:
        print("\nSkipping panel calibration (--skip-calibration set).")

    run_continuous_capture(base_url, args.duration, args.interval)


if __name__ == "__main__":
    main()