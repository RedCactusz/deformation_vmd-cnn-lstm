#!/bin/bash
# ============================================================================
# GMT: PREDICTED vs ACTUAL DEFORMATION MAP
# Unit: cm | Scale: Se0.005c
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/map_config.sh"
TARGET_DATE="${1:-unknown}"
MAP_NAME="map_predicted"

echo "GMT: PREDICTED vs ACTUAL  |  Target: $TARGET_DATE"
echo "Region: $REGION  Projection: $PROJECTION"

mkdir -p "$OUTPUT_DIR"

gmt begin "$OUTPUT_DIR/$MAP_NAME" pdf,png

gmt set MAP_FRAME_TYPE plain
gmt set MAP_FRAME_WIDTH 0.12c
gmt set MAP_TITLE_OFFSET 0.8c
gmt set FONT_TITLE "$FONT_TITLE"
gmt set FONT_LABEL "$FONT_LABEL"
gmt set FONT_ANNOT_PRIMARY "$FONT_ANNOT"


gmt basemap -R$REGION -J$PROJECTION -Bxa1 -Bya1 \
    -BWSne+t"GNSS Post-Seismic Deformation - Predicted vs Actual ($TARGET_DATE)"

gmt coast -R$REGION -J$PROJECTION -Df -W0.4p,dimgray -Slightblue -N1/0.3p -B

if [ -f "$GMT_INPUT_DIR/earthquake_event.gmt" ]; then
    gmt plot "$GMT_INPUT_DIR/earthquake_event.gmt" -R$REGION -J$PROJECTION \
        -Sa0.55c -Gred -W0.8p,darkred
fi

if [ -f "$GMT_INPUT_DIR/actual_vectors.gmt" ]; then
    gmt velo "$GMT_INPUT_DIR/actual_vectors.gmt" -R$REGION -J$PROJECTION \
        -Se0.005c/0.95 -A14p+e+a30 -W0.8p,red -Gred -L -N
fi

if [ -f "$GMT_INPUT_DIR/predicted_vectors.gmt" ]; then
    gmt velo "$GMT_INPUT_DIR/predicted_vectors.gmt" -R$REGION -J$PROJECTION \
        -Se0.005c/0.95 -A14p+e+a30 -W0.8p,forestgreen -Gforestgreen -L -N
fi

if [ -f "$GMT_INPUT_DIR/stations_coords.gmt" ]; then
    gmt plot "$GMT_INPUT_DIR/stations_coords.gmt" -R$REGION -J$PROJECTION \
        -Sc0.22c -Gblack -W0.4p,white
    gmt text "$GMT_INPUT_DIR/stations_coords.gmt" -R$REGION -J$PROJECTION \
        -F+f5.5p,Helvetica,black+jLM -Dj0.15c/0.15c
fi

gmt basemap -R$REGION -J$PROJECTION -Lg139.4/37.2+c38+w50k+l"50 km"+f
gmt basemap -R$REGION -J$PROJECTION -Tdg142.4/41.2+w1.2c+f2+l

cat > legend.txt << 'EOF'
H 12p Helvetica-Bold Legend
D 0.2c 1p
S 0.5c - 0.8c forestgreen 1.5p,forestgreen 0.5c Predicted (CNN-LSTM)
S 0.5c - 0.8c red 1.5p,red 0.5c Actual (GNSS)
S 0.5c a 0.28c red 0.8p,darkred 0.5c Epicenter (M9.0)
S 0.5c c 0.22c black 0.4p,white 0.5c GNSS Station
EOF
gmt legend legend.txt -R$REGION -J$PROJECTION \
    -DjTL+w5.0c+o0.3c -F+p0.5p+gwhite
rm -f legend.txt

gmt end

rm -f gmt.history topo.nc
echo "Done: $OUTPUT_DIR/$MAP_NAME"