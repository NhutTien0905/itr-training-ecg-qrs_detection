#! /bin/sh

DB_PATH=$1
REPORT_PATH=$2
OUTPUT_NAME=$3
EXT_REF=$4
EXT_AI=$5


OUTPUT="af"
cd $DB_PATH
echo "$DB_PATH"
for entry in *."$EXT_AI" ; do
      name=$(echo "$entry" | cut -f 1 -d '.')
       epicmp -r $name -a "$EXT_REF" "$EXT_AI" -L -A "$OUTPUT".out #sd.out
    done
sumstats "$OUTPUT".out >> "$OUTPUT_NAME".out

if [ ! -d "$REPORT_PATH" ]; then
    mkdir "$REPORT_PATH"
fi
mv "$OUTPUT_NAME".out "$REPORT_PATH"
