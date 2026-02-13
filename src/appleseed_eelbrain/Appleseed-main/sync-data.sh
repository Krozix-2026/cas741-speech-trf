# Use -ni to list changes
if [ ! -d "/Volumes/Seagate BarracudaFastSSD/Appleseed" ]
then
  echo "Data drive not connected"
  exit 0
fi
echo "scripts --> psych-linux1"
rsync -au --include={"appleseed/","setup.py"} --include={'*.py','*.txt'} --exclude='*' "$@" "." "cmb20003@psych-linux1.psy.uconn.edu:Code/Appleseed"
echo "data --> psych-linux1"
rsync -au --exclude={'.*'} --include={'eelbrain-cache/raw/*/* ica-20*','eelbrain-cache/raw/*/* 1-40*'} --exclude='eelbrain-cache/raw/*/*' --include={'eelbrain-cache/***','meg/***','mri/***','predictors/***'} --exclude='*' "$@" "/Volumes/Seagate BarracudaFastSSD/Appleseed/" "cmb20003@psych-linux1.psy.uconn.edu:Data/Appleseed/"
echo "data <-- psych-linux1"
rsync -au --exclude='.*' "$@" "cmb20003@psych-linux1.psy.uconn.edu:Data/Appleseed/eelbrain-cache" "/Volumes/Seagate BarracudaFastSSD/Appleseed"
