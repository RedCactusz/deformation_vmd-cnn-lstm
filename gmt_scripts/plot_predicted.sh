#!/bin/bash
# ============================================================================
# GMT Script: PREDICTED DEFORMATION MAP
# ============================================================================
# Visualisasi hasil prediksi model CNN-LSTM untuk periode mendatang

set -e

OUTPUT_DIR="outputs/plots"
GMT_INPUT_DIR="data/gmt_inputs"
MAP_NAME="map_predicted"

PROJECTION="M8i"
FRAME="a1g1"
REGION="139/147/37/45"
DPI="300"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "GMT Map Generation: PREDICTED DEFORMATION"
echo "=========================================="

gmt begin $MAP_NAME ps

gmt set MAP_FRAME_TYPE plain
gmt set FONT_LABEL 12p,Helvetica,black
gmt set FONT_ANNOT_PRIMARY 10p,Helvetica,black

# Main map
echo "Creating predicted deformation map..."
gmt basemap -R$REGION -J$PROJECTION -B$FRAME -BWSne+t"Predicted GNSS Deformation (CNN-LSTM)"

# Coastlines
gmt coast -R$REGION -J$PROJECTION -Slightyellow -W1p,black -N1

# Predicted vectors
if [ -f "$GMT_INPUT_DIR/predictions_coords.gmt" ]; then
    echo "Plotting predicted displacements..."
    gmt plot "$GMT_INPUT_DIR/predictions_coords.gmt" -R$REGION -J$PROJECTION \
        -Sc0.15c -Ggreen -W0.5p,darkgreen
fi

# Deformation grid (if available)
if [ -f "$GMT_INPUT_DIR/deformation_grid.gmt" ]; then
    echo "Plotting deformation grid..."
    gmt plot "$GMT_INPUT_DIR/deformation_grid.gmt" -R$REGION -J$PROJECTION \
        -Sc0.15c -Cgray -W0.1p
fi

# Grid
gmt basemap -R$REGION -J$PROJECTION -Bg1

# Scale
gmt basemap -R$REGION -J$PROJECTION -Lg140/38+c-7.5+w50k+l"Scale (km)"

# North arrow  
gmt basemap -R$REGION -J$PROJECTION -Tmg141/37+w2c+l

# Add legend
echo "Adding legend..."
cat > legend.txt << EOF
H 14 0 Legend
D 0 1p
S 0.4c c 0.4c green 0.5p,darkgreen 0.5c Predicted Displacement
S 0.4c c 0.2c gray 0.1p 0.5c Deformation Grid
EOF

gmt legend legend.txt -R$REGION -J$PROJECTION \
    -DjTR+w3c/3c+o0.3c -F+p0.5p+ggray

rm -f legend.txt

gmt end

ps2pdf $MAP_NAME.ps "$OUTPUT_DIR/$MAP_NAME.pdf"
convert -density $DPI -alpha off "$OUTPUT_DIR/$MAP_NAME.pdf" "$OUTPUT_DIR/$MAP_NAME.png"

rm -f $MAP_NAME.ps gmt.history

echo "=========================================="
echo "✓ Predicted deformation map created successfully"
echo "  Output: $OUTPUT_DIR/$MAP_NAME.pdf"
echo "          $OUTPUT_DIR/$MAP_NAME.png"
echo "=========================================="