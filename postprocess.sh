#/bin/sh

# suffix of target language files
lng=en

sed -r 's/\@\@ //g' | \
perl scripts/detruecase.perl | \
perl scripts/tokenizer.perl -l $lng
