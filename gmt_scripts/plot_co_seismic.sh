#!/bin/bash
# ============================================================================
# GMT Script: CO-SEISMIC DEFORMATION MAP
# ============================================================================
# Visualisasi pergeseran sesaat setelah gempa

set -e

PROJECT_NAME="gnss_deformation"
OUTPUT_DIR="outputs/plots"
GMT_INPUT_DIR="data/gmt_inputs"
MAP_NAME="map_co_seismic"

PROJECTION="M8i"
FRAME="a1g1"
REGION="139/147/37/45"
DPI="300"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "GMT Map Generation: CO-SEISMIC"
echo "=========================================="

gmt begin $MAP_NAME ps

gmt set MAP_FRAME_TYPE plain
gmt set FONT_LABEL 12p,Helvetica,black
gmt set FONT_ANNOT_PRIMARY 10p,Helvetica,black

# Main map
echo "Creating co-seismic deformation map..."
gmt basemap -R$REGION -J$PROJECTION -B$FRAME -BWSne+t"Co-seismic GNSS Deformation"

# Coastlines
gmt coast -R$REGION -J$PROJECTION -Slightblue -W1p,black -N1

# Earthquake epicenter marker
if [ -f "$GMT_INPUT_DIR/earthquake_event.gmt" ]; then
    echo "Marking earthquake epicenter..."
    gmt plot "$GMT_INPUT_DIR/earthquake_event.gmt" -R$REGION -J$PROJECTION \
        -Sa0.5c -Gred -W1p,darkred
fi

# Velocity vectors (co-seismic)
if [ -f "$GMT_INPUT_DIR/stations_velocity.gmt" ]; then
    echo "Plotting co-seismic displacement..."
    gmt velo "$GMT_INPUT_DIR/stations_velocity.gmt" -R$REGION -J$PROJECTION \
        -Sn1c/0.95 -A18p+e -W1p,darkred -L -N -D8 \
        -t40
fi

# Station points
if [ -f "$GMT_INPUT_DIR/stations_coords.gmt" ]; then
    echo "Plotting stations..."
    gmt plot "$GMT_INPUT_DIR/stations_coords.gmt" -R$REGION -J$PROJECTION \
        -Sc0.15c -Gdarkred -W0.5p,black
fi

# Grid
gmt basemap -R$REGION -J$PROJECTION -Bg1

# Scale
gmt basemap -R$REGION -J$PROJECTION -Lg140/38+c-7.5+w50k+l"Scale (km)"

# North arrow
gmt basemap -R$REGION -J$PROJECTION -Tmg141/37+w2c+l

gmt end

ps2pdf $MAP_NAME.ps "$OUTPUT_DIR/$MAP_NAME.pdf"
convert -density $DPI -alpha off "$OUTPUT_DIR/$MAP_NAME.pdf" "$OUTPUT_DIR/$MAP_NAME.png"

rm -f $MAP_NAME.ps gmt.history

echo "=========================================="
echo "✓ Co-seismic map created successfully"
echo "  Output: $OUTPUT_DIR/$MAP_NAME.pdf"
echo "          $OUTPUT_DIR/$MAP_NAME.png"
echo "=========================================="