# Use -ni to list changes
if [ ! -d "/Volumes/Seagate BarracudaFastSSD/Dutch American EEG" ]
then
  echo "Data drive not connected"
  exit 0
fi

#echo "alien <-- here"
#rsync -au --exclude='Gammatone' --exclude='eelbrain Burgundy.log' --exclude='.*' "$@" ~/Data/Burgundy/ christian@137.99.35.165:Data/Burgundy/

#echo "alien --> here"
#rsync -au --exclude='.*' "$@" christian@137.99.35.165:Data/Burgundy/eelbrain-cache ~/Data/Burgundy

# One-way sync for TRFs only (assume they are merged remotely)
# After merging TRFs remotely, run with --delete flag
echo "alien --> here:  cache"
rsync -au --exclude='.*' --exclude='trf-backup' --exclude='eelbrain-cache/model-test' "$@" christian@137.99.35.165:Data/Burgundy/eelbrain-cache ~/Data/Burgundy

# One-way sync for predictors
#echo "alien --> here:  predictors"
#rsync -au --delete --exclude='.*' "$@" christian@137.99.35.165:Data/Burgundy/predictors ~/Data/Burgundy
