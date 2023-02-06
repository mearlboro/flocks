#/bin/sh

indir="img"
outdir="gif"

if [ ! -d "$indir" ]; then
    echo "Script expects a directory called $indir to pull simulations from"
    exit
fi

if [ ! -d "$outdir" ]; then
    mkdir -p -v $outdir
fi


for dir in $indir/*; do
    [ -d "$dir" ] || continue;

    name=$(basename $dir)
    first_file=`find $dir -type f | sort | head -n 1`
    start_at=`echo $(basename $first_file) | tr -cd '[0-9]'`

    echo "Creating gif from $dir to $outdir/$name.gif starting at $start_at"

    # create a pallette for GIF compression
    ffmpeg -y -i "$first_file" -vf palettegen "$dir/palette.png"

    ffmpeg -framerate 12 -start_number "$start_at" -i "$dir"/%d.jpg -i "$dir/palette.png" -lavfi paletteuse -r 24 -s 800x600 "$outdir/$name.gif"

    rm "$dir/palette.png"
done
