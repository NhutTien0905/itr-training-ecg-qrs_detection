#! /bin/sh


DB_PATH=$1
REPORT_PATH=$2
DEL_RES=$3

cd $DB_PATH
echo "remove all temporary files in $DB_PATH"

rm `ls -I "*.atr" -I "*.dat" -I "*.air" -I "*.qrs*" -I "*.hea"`

#if  [ $DEL_RES = "1" ]; then
#    rm *.ai*
#fi

if [ ! -d "$REPORT_PATH" ]; then
    mkdir "$REPORT_PATH"
fi
