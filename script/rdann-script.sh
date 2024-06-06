#! /bin/sh
DB_PATH=$1
EXT_REF=$2
EXT_AI=$3
EXT_HR=$4

cd $DB_PATH
echo "$DB_PATH"

for entry in *."$EXT_AI" ; do
      name=$(echo "$entry" | cut -f 1 -d '.')
      echo "$name"
      rdann -r $name -a "$EXT_REF" >"$name.rd$EXT_REF"
      rdann -r $name -a "$EXT_AI" >"$name.rd$EXT_AI"
    done


