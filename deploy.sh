#!/bin/bash

DIR="$(dirname "$(realpath "$0")")"

/opt/homebrew/bin/python3 -m ansible playbook -i "$DIR/inventory.ini" "$DIR/playbook.yml" -u cnimmo
