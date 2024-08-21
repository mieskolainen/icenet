#!/bin/sh
#
# Clear all output

echo "Cleaning output, figs, checkpoint and tmp folders completely (all results gone) -- Are you sure? (write yes or no)"
read yn
case $yn in
    Yes|yes|Y|y )
        echo "Deleting ..."
        rm -f -r output/* 
        rm -f -r figs/* 
        rm -f -r checkpoint/* 
        rm -f -r tmp/*
        ;;
    No|no|N|n ) 
        echo "Aborted."
        ;;
    * ) 
        echo "Invalid input. Aborted."
        ;;
esac
