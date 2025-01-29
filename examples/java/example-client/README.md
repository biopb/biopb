## Example java implementation
This is a minimal java client to query biopb.image server.

The main function will read a image file, query the lacss.biopb.org server, and print out the deteted cell boundaries. 

### Steps to run this example

1. copy or link the relavent protocol files to src/main/proto/
```
cp -r ../../../biopb src/main/proto/
```
2. build and run with maven
```
mvn exec:java -Dexec.mainClass=org.biopb.image.example.Client -Dexec.args="<image-file-path>"
```
