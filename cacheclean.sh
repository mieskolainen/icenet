#!/bin/sh
#
# Clear output cache

echo "Cleaning output cache folder -- Are you sure? (write yes or no)"
read yn
case $yn in
    Yes|yes|Y|y )
        echo "Deleting ..."
        rm -f -r output/*
        ;;
    No|no|N|n ) 
        echo "Aborted."
        ;;
    * ) 
        echo "Invalid input. Aborted."
        ;;
esac
