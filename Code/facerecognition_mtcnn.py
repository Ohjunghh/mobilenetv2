import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
import time

# MobileNetV2 모델 불러오기 및 커스터마이징
trained_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights=None)
trained_model.trainable = True

last_layer_output = trained_model.get_layer('out_relu').output
x = GlobalAveragePooling2D()(last_layer_output)
x = Dropout(0.8)(x)
x = Dense(120)(x)
output = Dense(105, activation="softmax")(x)
model_mobilenet = Model(trained_model.input, output)

# 모델 가중치 로드
model_mobilenet.load_weights('model_mobilenet.h5')

# 마지막 분류 레이어 제거
model_mobilenet_facenet = Model(model_mobilenet.input, model_mobilenet.get_layer('dense_1').output)

def calculate_embeddings(path, model, input_shape):
    embeddings = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            img = cv2.imread(img_path)
            img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
            img = tf.expand_dims(img, axis=0)
            img_predict = model.predict(img)
            emb_item = [img_predict, folder]
            embeddings.append(emb_item)
    return embeddings

# 데이터셋 임베딩 계산
embeddings = calculate_embeddings("images/", model_mobilenet_facenet, (160, 160))

# MTCNN 얼굴 탐지기 로드
detector = MTCNN()

# 웹캠에서 비디오 캡처
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, img = cap.read()
    if not ret:
        break

    start_time = time.time()  # Start time for FPS calculation
    
    # 얼굴 검출
    faces = detector.detect_faces(img)

    # 각 얼굴에 대해
    for face in faces:
        x, y, w, h = face['box']
        cam_img = img[y:y+h, x:x+w]
        cam_img = cv2.resize(cam_img, (160, 160), interpolation=cv2.INTER_LINEAR)
        cam_img = tf.expand_dims(cam_img, axis=0)
        img_predict_2 = model_mobilenet_facenet.predict(cam_img)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        maxVal = 0.0
        best_match = "Not Recognized"
        
        # 임베딩 벡터와의 유사도 비교
        for embedding in embeddings:
            val = cosine_similarity(embedding[0], img_predict_2)[0][0]
            if val > maxVal:
                maxVal = val
                best_match = embedding[1]

        if maxVal < 0.8:
            best_match = "Not Recognized"
        
        cv2.putText(img, f"{best_match} ({maxVal:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 표시
    cv2.imshow('img', img)
    # ESC 키를 눌러 종료
    if cv2.waitKey(30) & 0xff == 27:
        break
        
# 비디오 캡처 객체 해제
cap.release()
cv2.destroyAllWindows()
