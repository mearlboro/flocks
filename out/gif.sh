#/bin/sh

indir="img"
outdir="gif"

if [ ! -d "$indir" ]; then
    echo "Script expects a directory called $indir to pull simulations from"
    exit
fi

if [ ! -d "$outdir"]; then
    mkdir -p -v $outdir
fi

for dir in $indir/*; do
    [ -d "$dir" ] || continue

    name=$(basename $dir)

    echo "Creating gif from $dir to $outdir/$name.gif"
    ffmpeg -framerate 12 -i $dir/%d.jpg  -r 24 $outdir/$name.gif
done
