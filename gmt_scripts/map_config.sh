#!/bin/bash
# ============================================================================
# GMT Shared Configuration — Paper-Quality GNSS Deformation Maps
# ============================================================================
# Source: source "$(dirname "$0")/map_config.sh"

OUTPUT_DIR="outputs/plots"
GMT_INPUT_DIR="data/gmt_inputs"

REGION="139.2/142.6/37.0/41.6"
PROJECTION="M18c"
FRAME="a1g1"
DPI="300"

# Common styling
FONT_TITLE="14p,Helvetica-Bold,black"
FONT_LABEL="11p,Helvetica,black"
FONT_ANNOT="9p,Helvetica,black"
FONT_LEGEND="9p,Helvetica,black"

mkdir -p "$OUTPUT_DIR"
