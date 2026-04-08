'''
Sanity check to see if the unidentified
markers are passed on to the system.
'''

import qtm
import asyncio

def on_packet(packet):
    # 1. Retrieve identified markers (Labeled)
    # Note: 'markers' is a list of points; 'info' contains component metadata
    info, markers = packet.get_3d_markers()
    
    # 2. Retrieve unidentified trajectories (Unlabeled)
    # Ensure "3dnolabels" is included in the components during setup
    info_un, markers_un = packet.get_3d_markers_no_label()
    
    # 3. Calculate marker counts
    num_labeled = len(markers) if markers else 0
    num_unlabeled = len(markers_un) if markers_un else 0
    
    # 4. Access the frame number directly from the packet object
    frame = packet.framenumber
    
    if num_labeled + num_unlabeled > 0:
        print(f"Frame {frame}: Labeled={num_labeled}, Unlabeled={num_unlabeled}")
        
        # Print coordinates of the first unlabeled marker for verification
        if num_unlabeled > 0:
            m = markers_un[0]
            print(f"  [Unlabeled] ID: {m.id} -> x: {m.x:.2f}, y: {m.y:.2f}, z: {m.z:.2f}")
    else:
        print(f"Frame {frame}: No markers detected.")

async def setup():
    print("Connecting to QTM...")
    connection = await qtm.connect("192.168.1.122")
    if connection is None:
        print("Connection failed.")
        return

    print("Connected successfully. Starting stream...")
    # Essential: Include "3dnolabels" to receive the unidentified trajectories
    await connection.stream_frames(components=["3d", "3dnolabels"], on_packet=on_packet)
    
    # Keep the event loop alive to continue receiving packets
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    # Execute the asynchronous entry point using asyncio
    try:
        asyncio.run(setup())
    except KeyboardInterrupt:
        print("\nStopped by user.")