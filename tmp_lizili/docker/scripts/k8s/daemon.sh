#!/bin/bash
TTL=86400
tick=$TTL
while [ $tick -gt 0 ]; do
  if [ ! -z "$(w | grep 'pts/')" ]; then
    tick=$TTL
  fi
  tick=$(( $tick - 1 ))
  echo "tick" $tick
  sleep 1
done
