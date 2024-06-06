#! /bin/sh

DB_PATH=$1
REPORT_PATH=$2
EXT_REF=$3
EXT_AI=$4
OUTPUT_NAME=$5
OUTPUT_NAME2=$6

OUTPUT="bxb"
cd $DB_PATH
echo "$DB_PATH"
for entry in *."$EXT_AI" ; do
      name=$(echo "$entry" | cut -f 1 -d '.')
      echo "bxb $name"
      bxb -r $name -a "$EXT_REF" "$EXT_AI" -L "$OUTPUT".out sd.out
      bxb -r $name -a "$EXT_REF" "$EXT_AI" -S "$OUTPUT_NAME2".out
    done
sumstats "$OUTPUT".out >> "$OUTPUT_NAME".out

if [ ! -d "$REPORT_PATH" ]; then
    mkdir "$REPORT_PATH"
fi
mv "$OUTPUT_NAME".out "$REPORT_PATH"
mv "$OUTPUT_NAME2".out "$REPORT_PATH"
