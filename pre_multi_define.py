import time
import json
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# List the Crazyflie indices you want to configure
CF_INDICES = [1, 2, 3]

def load_cf_configs(cf_indices):
    """Load configs from config_crazyflie_*.json and build URI->marker_ids mapping."""
    configs = {}
    for cf_idx in cf_indices:
        cfg_path = f"config_crazyflie_{cf_idx}.json"
        with open(cfg_path, "r") as f:
            spec = json.load(f)

        uri = spec["URI"]
        # Generate marker IDs like [11,12,13,14], [21,22,23,24], ...
        marker_ids = [int(f"{cf_idx}{i}") for i in range(1, 5)]
        configs[uri] = marker_ids

    return configs

def set_active_marker_ids(uri, ids):
    """Connect to a Crazyflie and set active marker IDs."""
    print(f"Connecting {uri} and setting marker IDs...")
    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
            scf.cf.param.set_value("activeMarker.front", str(ids[0]))
            scf.cf.param.set_value("activeMarker.right", str(ids[1]))
            scf.cf.param.set_value("activeMarker.back", str(ids[2]))
            scf.cf.param.set_value("activeMarker.left", str(ids[3]))

            # Ensure Active Marker mode
            scf.cf.param.set_value("activeMarker.mode", "1")

            print(f"Success: {uri} IDs set to {ids}")
            time.sleep(0.5)

    except Exception as e:
        print(f"Failed to connect or set {uri}: {e}")

if __name__ == "__main__":
    cflib.crtp.init_drivers()

    cf_configs = load_cf_configs(CF_INDICES)

    for uri, marker_ids in cf_configs.items():
        set_active_marker_ids(uri, marker_ids)

    print("All Crazyflie marker IDs configured. You can define rigid bodies in QTM now.")
