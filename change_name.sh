#!/bin/bash 

sed -i 's/magma/thunder/g' include/thunder/*/*
sed -i 's/magma/thunder/g' src/*/*
sed -i 's/magma/thunder/g' src/*
sed -i 's/magma/thunder/g' test/*
sed -i 's/magma/thunder/g' test/*/*


sed -i 's/MAGMA/THUNDER/g' include/thunder/*/*
sed -i 's/MAGMA/THUNDER/g' src/*
sed -i 's/MAGMA/THUNDER/g' src/*/*
sed -i 's/MAGMA/THUNDER/g' test/*
sed -i 's/MAGMA/THUNDER/g' test/*/*

sed -i 's/MagMA/Thunder/g' include/thunder/*/*
sed -i 's/MagMA/Thunder/g' src/*/*
sed -i 's/MagMA/Thunder/g' src/*
sed -i 's/MagMA/Thunder/g' test/*
sed -i 's/MagMA/Thunder/g' test/*/*