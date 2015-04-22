
# Matlab binary path
MATLAB_BIN="/Applications/MATLAB_R2012a.app/bin/matlab"

# Java classpath (according to the folder structure, do not modify) 
LIBPATH="./lib/weka-3.7.6.jar:./lib/mulan.jar"
CLASSPATH="./src/FMC.jar"

# Temporal files will be stored here, change only if necessary
OUTPATH="./temp_files/"
mkdir $OUTPATH


# The pipeline starts here. First it runs the java application with the specified parameters,
# then it runs the matlab step.
echo "Starting Training phase"

java -Xms2000m -Xmx2000m -cp $LIBPATH:$CLASSPATH es.jarias.FMC.RunFMC -outpath $OUTPATH $@

# Count number of folds to input the matlab program:
FOLDS=`ls $OUTPATH | grep conf | wc -l`
if [ $FOLDS -eq 0 ]
then
    exit
fi

echo
echo "Loading Matlab..."
echo

$MATLAB_BIN -nodisplay -nosplash -nodesktop -r "cd src;MRFInference($FOLDS, '../$OUTPATH', 'approximate');exit;"
rm $OUTPATH/*.txt

# END
