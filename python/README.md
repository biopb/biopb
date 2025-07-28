# BIOPB python package

A simple utility package to simplify usage of biopb in python.

 - Create a packge to organize the protocol genfiles.
 - Provide utility functions for convert to/from numpy data type.


## Installation

```sh
pip install biopb
```

## Build and install locally
```
./build.sh
pip install -e .
```

## Example usage
``` python
import grpc
import biopb.image as proto
from biopb.image.utils import serialize_from_numpy, deserialize_to_numpy

with grpc.insecure_channel(SERVER) as channel:
    response = proto.ProcessImageStub(channel).Run( 
        proto.ProcessRequest( 
            image_data=proto.ImageData(
                pixels=serialize_from_numpy(image)
            )
        )
    )

result = deserialize_to_numpy(response.image_data.pixels)
```
