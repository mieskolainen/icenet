#!/bin/sh
#

echo "Cleaning output, figs, checkpoint and tmp folders completely (all results gone) -- Are you sure?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) rm -f -r output/*; rm -f -r figs/*; rm -f -r checkpoint/*; rm -f -r tmp/*; break;;
        No ) break;;
    esac
done
