#! /bin/sh


DB_PATH=$1
EXT_OUT=$2
EXT_INT=$3

cd $DB_PATH
echo "$DB_PATH"

for entry in *."$EXT_INT" ; do
      name=$(echo "$entry" | cut -f 1 -d '.')
      echo "$name"
      wrann -r $name -a "$EXT_OUT" <"$name.$EXT_INT"
    done


