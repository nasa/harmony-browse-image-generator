#!/bin/bash
###############################################################################
#
# A bash script to extract only the notes related to the most recent version of
# HyBIG from CHANGELOG.md
#
# 2023-06-16: Created.
# 2023-10-10: Copied from earthdata-varinfo repository to HOSS.
# 2024-01-03: Copied from HOSS repository to the Swath Projector.
# 2024-01-23: Copied and modified from Swath Projector repository to HyBIG.
#
###############################################################################

CHANGELOG_FILE="CHANGELOG.md"
VERSION_PATTERN="^## v"

# Read the file and extract text between the first two occurrences of the
# VERSION_PATTERN
result=$(awk "/$VERSION_PATTERN/{c++; if(c==2) exit;} c==1" "$CHANGELOG_FILE")

# Print the result
echo "$result" |  grep -v "$VERSION_PATTERN"
