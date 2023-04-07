#!/bin/sh
#

echo "Cleaning output, figs, checkpoint and tmp folders completely (all results gone) -- Are you sure?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) rm output/* -f -r; rm figs/* -f -r; rm checkpoint/* -f -r; rm tmp/* -f -r; break;;
        No ) break;;
    esac
done
