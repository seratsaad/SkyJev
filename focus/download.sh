#!/bin/bash
# Fetch the chosen public WFI through-focus frames from the ESO archive and decompress them.
mkdir -p "$(dirname "$0")/data" && cd "$(dirname "$0")/data"
while read id; do
  f="${id//:/-}.fits"
  [ -s "$f" ] && continue
  curl -s -L --max-time 300 -o "$f.Z" "https://dataportal.eso.org/dataPortal/file/$id" && uncompress -f "$f.Z" 2>/dev/null && echo "ok $id" || { echo "FAIL $id"; rm -f "$f.Z"; }
done < ../download_list.txt
echo "DOWNLOAD DONE"
