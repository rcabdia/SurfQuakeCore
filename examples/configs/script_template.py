import numpy as np

"""
This script is an example of how to plug your own script into the surfquake workflow. 
Important: Main method must be called always run!
Commands quick and processing daily uses stream and optionally inventory, processing can also optionally use event

For python advanced users: Even though this is a simple method with inputs stream, inventory and event. One can design
different objects that interacts inside this script. For example one can import a class and then be used 
inside the "run" method.

"""
def run(stream, **kwargs):

    '''
    User-defined post-processing hook for each event.

    :param stream: obspy Stream
    :param inventory: obspy Inventory (passed via kwargs)
    :param event: dict with keys like origin_time, latitude, longitude, depth and magnitude (passed via kwargs)
    '''

    inventory = kwargs.pop('inventory', None)
    event = kwargs.pop('event', None)

    if event is not None:
        print(f"Post-processing {len(stream)} traces from event at {event['origin_time']}")
        print(event["origin_time"], event["latitude"], event["longitude"], event["depth"], event["magnitude"])
    else:
        print(f"Post-processing {len(stream)} traces (no event info provided)")

    if inventory is not None:
        print(inventory)

    # Example: calculate global mean amplitude
    all_data = np.hstack([tr.data for tr in stream])
    print("Mean amplitude:", np.mean(all_data))

    # Example: apply a simple differencing operator to each trace
    for tr in stream:
        tr.data = np.diff(tr.data)

    return stream