#! /bin/sh

# Parse arguments
IMAGE=""
DEBUG="no"  # Default to off for non-plugin code

for arg in "$@"; do
    case "$arg" in
        --debug)
            DEBUG="yes"
            ;;
        --no-debug)
            DEBUG="no"
            ;;
        *)
            IMAGE="$arg"
            ;;
    esac
done

.venv/bin/python -c "
import logging
import napari

# Always enable debug for biopb-mcp plugin
logging.getLogger('biopb_mcp').setLevel(logging.DEBUG)

# Optionally enable debug for rest of code
if '$DEBUG' == 'yes':
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig()

viewer = napari.Viewer()

# Preload image if provided
if '$IMAGE':
    viewer.open('$IMAGE')

napari.run()
"
