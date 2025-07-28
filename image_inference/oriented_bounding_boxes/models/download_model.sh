
name="YOLO_v8_small_Oriented_Bounding_Boxes_1024_1024_3_onnx"
wget https://developer.memryx.com/model_explorer/2p0/${name}.zip
unzip ${name}.zip -d models
mv models/${name}.dfp models/yolov8s-obb.dfp
mv models/${name}_post.onnx models/yolov8s-obb_post.onnx
rm ${name}.zip;

