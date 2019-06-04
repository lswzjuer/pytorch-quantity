#!/bin/bash

for line in $(cat scripts/release_path); do
  if [[ ! -z $(find modules/$line 2>&1 | grep '\.cc') ]]; then
    echo "Directory containing .cc or .h files in release_path!"
    echo "check $line"
    exit 1
  fi
done
exit 0
