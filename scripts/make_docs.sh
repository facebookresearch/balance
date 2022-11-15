#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

# Exit if any error occurs
set -e

usage() {
  echo "Usage: $0"
  echo ""
  echo "Build balance documentation. Must be executed from root of balance repository."
  echo ""
  exit 1
}

while getopts 'h:' flag; do
  case "${flag}" in
    h)
      usage
      ;;
    *)
      usage
      ;;
  esac
done

echo "-----------------------------------"
echo "Generating API reference via Sphinx"
echo "-----------------------------------"
cd sphinx || exit
make html
cd .. || exit

echo "-----------------------------------"
echo "Getting Docusaurus deps"
echo "-----------------------------------"
cd website || exit
yarn

echo "--------------------------------------------"
echo "Moving Sphinx artifacts to Docusaurus"
echo "--------------------------------------------"
cd ..
mkdir -p "website/static/api_reference/"
cp -r sphinx/_build/* website/static/api_reference/ || exit
cd website || exit

echo "-----------------------------------"
echo "Building static Docusaurus site"
echo "-----------------------------------"
yarn build
