rem if exist "./src/caffe/proto/caffe.pb.h" (
rem    echo caffe.pb.h remains the same as before
rem ) else (
echo caffe.pb.h is being generated
"./tools/protoc" -I="./src/caffe/proto" --cpp_out=dllexport_decl=CAFFE_DLL_EXPORT:"./src/caffe/proto" "./src/caffe/proto/crfasrnn_caffe.proto"
"./tools/protoc" -I="./src/caffe/proto" --cpp_out=dllexport_decl=CAFFE_DLL_EXPORT:"./src/caffe/proto" "./src/caffe/proto/crfasrnn_caffe_pretty_print.proto"
md "./include/caffe/proto/"
move ".\src\caffe\proto\caffe.pb.h" ".\include\caffe\proto\"
rem )
rem  protoc --cpp_out=dllexport_decl=MY_EXPORT_MACRO:path/to/output/dir myproto.proto
