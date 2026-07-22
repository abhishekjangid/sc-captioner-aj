from llamafactory.train.sc.yolo_verifier import YoloObjectVerifier, verify_object
print(verify_object("/Users/rutvika/source_code/dataset/coco/coco6k/000000000071.jpg", "bird"))
print(verify_object("/Users/rutvika/source_code/dataset/docci/test_00475.jpg", "bird"))

print(YoloObjectVerifier._normalize_object_name("TV-SCREEN"))
print(YoloObjectVerifier._lemmatize_label("cars"))
print(YoloObjectVerifier._lemmatize_label("books"))