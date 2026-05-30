#!/bin/bash
# ============================================================================
# GMT Script: PRE-SEISMIC DEFORMATION MAP
# ============================================================================
# Visualisasi vektor deformasi GNSS sebelum gempa menggunakan GMT 6

set -e  # Exit on error

# Configuration
PROJECT_NAME="gnss_deformation"
OUTPUT_DIR="outputs/plots"
GMT_INPUT_DIR="data/gmt_inputs"
MAP_NAME="map_pre_seismic"

# GMT parameters
PROJECTION="M8i"                    # Mercator projection, 8 inch width
FRAME="a1g1"                        # Frame and grid
COASTLINE="black"
BORDER="thicker"
DPI="300"

# Region (Indonesia - Jawa region)
REGION="139/147/37/45"             # Japan Region (Tohoku)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "GMT Map Generation: PRE-SEISMIC"
echo "=========================================="
echo "Region: $REGION"
echo "Projection: $PROJECTION"
echo "Output: $OUTPUT_DIR/$MAP_NAME"
echo "=========================================="

# Initialize GMT session
gmt begin $MAP_NAME ps

# Set GMT defaults
gmt set MAP_FRAME_TYPE plain
gmt set FONT_LABEL 12p,Helvetica,black
gmt set FONT_ANNOT_PRIMARY 10p,Helvetica,black
gmt set MAP_TITLE_OFFSET 0.5c
gmt set MAP_FRAME_WIDTH 0.1c

# Main map
echo "Creating base map..."
gmt basemap -R$REGION -J$PROJECTION -B$FRAME -BWSne+t"Pre-Seismic GNSS Deformation"

# Add coastlines
echo "Adding coastlines..."
gmt coast -R$REGION -J$PROJECTION -Slightblue -W1p,$COASTLINE -N1

# Add gridlines
echo "Adding grid..."
gmt basemap -R$REGION -J$PROJECTION -Bg1

# Plot velocity vectors (if file exists)
if [ -f "$GMT_INPUT_DIR/velocity_pre_seismic.gmt" ]; then
    echo "Plotting velocity vectors..."
    gmt velo "$GMT_INPUT_DIR/velocity_pre_seismic.gmt" -R$REGION -J$PROJECTION \
        -Sw -A18p+e -W0.5p,red -L -N \
        -t40
fi

# Plot earthquake epicenter
if [ -f "$GMT_INPUT_DIR/earthquake_event.gmt" ]; then
    echo "Marking earthquake epicenter..."
    gmt plot "$GMT_INPUT_DIR/earthquake_event.gmt" -R$REGION -J$PROJECTION \
        -Sa0.5c -Gred -W1p,darkred
fi

# Plot station locations
if [ -f "$GMT_INPUT_DIR/stations_coords.gmt" ]; then
    echo "Plotting station locations..."
    gmt plot "$GMT_INPUT_DIR/stations_coords.gmt" -R$REGION -J$PROJECTION \
        -Sc0.15c -Gblue -W0.5p,darkblue
    
    # Add station labels
    gmt text "$GMT_INPUT_DIR/stations_coords.gmt" -R$REGION -J$PROJECTION \
        -F+f7p,Helvetica,black+jLM -Dj0.1c/0.1c
fi

# Add scale
echo "Adding scale..."
gmt basemap -R$REGION -J$PROJECTION -Lg140/38+c-7.5+w50k+l"Scale (km)"

# Add north arrow
echo "Adding north arrow..."
gmt basemap -R$REGION -J$PROJECTION -Tmg141/37+w2c+l

# Add color bar for deformation (if applicable)
# gmt colorbar -Ccolors.cpt -Dx8c/-1c+w10c/0.5c+h -Bxaf -By+lDeformation

# Finalize
gmt end

# Convert to PDF and PNG
echo "Converting to multiple formats..."
ps2pdf $MAP_NAME.ps "$OUTPUT_DIR/$MAP_NAME.pdf"
convert -density $DPI -alpha off "$OUTPUT_DIR/$MAP_NAME.pdf" "$OUTPUT_DIR/$MAP_NAME.png"

# Cleanup
rm -f $MAP_NAME.ps gmt.history

echo "=========================================="
echo "✓ Pre-seismic map created successfully"
echo "  Output: $OUTPUT_DIR/$MAP_NAME.pdf"
echo "          $OUTPUT_DIR/$MAP_NAME.png"
echo "=========================================="