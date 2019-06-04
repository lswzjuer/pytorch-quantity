RESULT=$(clang-tidy "$@" 2>&1)
RET=$?
[ $RET -eq 0 ] || echo "$RESULT" && exit $RET
