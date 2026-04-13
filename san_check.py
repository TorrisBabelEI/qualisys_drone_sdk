'''
Sanity check to track 6DOF Rigid Bodies with explicit Name discovery.
'''

import qtm
import asyncio

# Global variable to store body names retrieved from settings
body_names = []

def on_packet(packet):
    global body_names
    # 1. Get 6DOF Body data
    info, bodies = packet.get_6d()
    
    frame = packet.framenumber
    
    if bodies:
        num_bodies = len(bodies)
        print(f"\nFrame {frame}: Total Bodies Tracked = {num_bodies}")
        
        # Define the target list from your crazyflies.yaml
        targets = ["cf_01", "cf_02", "cf_03", "cf_04", "cf_05"]
        
        for target in targets:
            if target in body_names:
                idx = body_names.index(target)
                # Check if the body index is within the received packet data
                if idx < len(bodies):
                    pos, rot = bodies[idx]
                    # Position is in mm; convert to meters for ROS
                    print(f"  [FOUND] {target} -> x: {pos.x/1000:.3f}[m], y: {pos.y/1000:.3f}[m], z: {pos.z/1000:.3f}[m]")
            else:
                print(f"  [MISSING] {target} (Current QTM Names: {body_names})")
    else:
        print(f"Frame {frame}: No 6D data in packet.")

async def setup():
    global body_names
    print("Connecting to QTM at 192.168.1.122...")
    connection = await qtm.connect("192.168.1.122")
    if connection is None:
        print("Connection failed.")
        return

    # 2. CRITICAL: Fetch the parameters/settings to get the actual Body Names
    # This allows the script to map the index in the packet to a name like 'cf_03'
    params_xml = await connection.get_parameters(parameters=["6d"])
    import xml.etree.ElementTree as ET
    root = ET.fromstring(params_xml)
    body_names = [body.find("Name").text for body in root.findall(".//Body")]
    
    print(f"Discovered Body Names in QTM: {body_names}")
    print("Starting 6D Rigid Body stream...")
    
    await connection.stream_frames(components=["6d"], on_packet=on_packet)
    
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(setup())
    except KeyboardInterrupt:
        print("\nStopped by user.")