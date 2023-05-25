#! /usr/bin/bash
SRC=/aicenter2/QSAR_DATA/Data/RAW_Data
DEST=/run/user/1002
echo copy from ${SRC}/fsmol.tar to ${DEST}/fsmol.tar
cp ${SRC}/fsmol.tar ${DEST}/fsmol.tar
echo untar ${DEST}/fsmol.tar
tar -xf ${DEST}/fsmol.tar -C ${DEST}/
echo copy from ${SRC}/fs-mol-debug to ${DEST}/fs-mol-debug
cp -r ${SRC}/fs-mol-debug ${DEST}/fs-mol-debug
mkdir ${DEST}/tmp
echo copy from ${SRC}/fsmol_chembl32 to ${DEST}/tmp
cp -r ${SRC}/fsmol_chembl32 ${DEST}/tmp
