#!/usr/bin/env bash
# Run one step of the TinyGPT journey.
#
#   ./run.sh 1                    run Step01 (any number 1-11)
#   ./run.sh train                train the tiny GPT and save it (step 12)
#   ./run.sh generate "Hello"     let the saved model write text
#   ./run.sh finetune [medium]    teach GPT-2 to follow instructions (book chapter 7)
#   ./run.sh chat [small|medium]  chat with the fine-tuned GPT-2
set -euo pipefail
cd "$(dirname "$0")"

export MAVEN_OPTS="${MAVEN_OPTS:-} --enable-native-access=ALL-UNNAMED"

case "${1:-}" in
  ""|-h|--help)
    echo "Usage: ./run.sh <1-12 | train | generate \"prompt\" | finetune | chat>"
    echo
    ls src/main/java/com/tinygpt/examples | sed 's/\.java$//; s/^/  /'
    exit 0
    ;;
  train)
    main=com.tinygpt.TinyGPT
    args=""
    ;;
  generate)
    main=com.tinygpt.TinyGPT
    args="generate '${2:-Every effort moves you}'"
    ;;
  finetune)
    main=com.tinygpt.FineTune
    args="${2:-small}"
    ;;
  chat)
    # Run Java directly (not through Maven) so typing in the terminal works normally
    ./mvnw -q compile dependency:build-classpath -Dmdep.outputFile=target/classpath.txt 2>&1 | grep -v -E '^WARNING: ' || true
    exec java --enable-native-access=ALL-UNNAMED -cp "target/classes:$(cat target/classpath.txt)" com.tinygpt.Chat ${2:-} \
      2> >(grep -v -E 'LegacyTypeDispatch' >&2)
    ;;
  *)
    step=$(printf "%02d" "$1")
    file=$(ls src/main/java/com/tinygpt/examples/Step${step}_*.java 2>/dev/null | head -1 || true)
    if [ -z "$file" ]; then
      echo "No step $1. Try ./run.sh --help"
      exit 1
    fi
    main="com.tinygpt.examples.$(basename "$file" .java)"
    args=""
    ;;
esac

./mvnw -q compile exec:java -Dexec.mainClass="$main" -Dexec.args="$args" 2>&1 \
  | grep --line-buffered -v -E '^WARNING: |LegacyTypeDispatch'
