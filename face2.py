import cv2
import numpy as np
from os import listdir
from os.path import isdir, isfile, join

# 얼굴 인식용 haar/cascade 로딩
face_classifier = cv2.CascadeClassifier('./haarcascade_frontface.xml')    

mosaic_rate = 20

# 사용자 얼굴 학습
def train(name):
    data_path = 'faces/' + name + '/'
    #파일만 리스트로 만듬
    face_pics = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    
    Training_Data, Labels = [], []
    
    for i, files in enumerate(face_pics):
        image_path = data_path + face_pics[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 이미지가 아니면 패스
        if images is None:
            continue    
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    if len(Labels) == 0:
        print("There is no data to train.")
        return None
    Labels = np.asarray(Labels, dtype=np.int32)
    # 모델 생성
    model = cv2.face.LBPHFaceRecognizer_create()
    # 학습
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print(name + " : Model Training Complete!!!!!")

    #학습 모델 리턴
    return model

# 여러 사용자 학습
def trains():
    #faces 폴더의 하위 폴더를 학습
    data_path = 'faces/'
    # 폴더만 색출
    model_dirs = [f for f in listdir(data_path) if isdir(join(data_path,f))]
    
    #학습 모델 저장할 딕셔너리
    models = {}
    # 각 폴더에 있는 얼굴들 학습
    for model in model_dirs:
        print('model :' + model)
        # 학습 시작
        result = train(model)
        # 학습이 안되었다면 패스!
        if result is None:
            continue
        # 학습되었으면 저장
        print('model2 :' + model)
        models[model] = result

    # 학습된 모델 딕셔너리 리턴
    return models    

#얼굴 검출
def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        if faces is():
            return img,[],[],[],[],[]
        for(x,y,w,h) in faces:
            #cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))

        for(x, y, w, h) in faces:
            try:
                # 얼굴 부분 자르기
                face_img = img[x:y+h, x:x+w]
                # 자른 이미지를 지정한 배율로 확대/축소하기
                face_img = cv2.resize(face_img, (w//mosaic_rate, h//mosaic_rate))
                # 확대/축소한 그림을 원래 크기로 돌리기
                face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA)
                # 원래 이미지에 붙이기
                #img[y:y+h, x:x+h] = face_img

            except Exception as e:
                print(str(e))
        
        return img,roi,face_img,h,y,x   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달

# 인식 시작
def run(models):    
    #카메라 열기 
    cap = cv2.VideoCapture(0)
    
    while True:
        #카메라로 부터 사진 한장 읽기 
        ret, frame = cap.read()
        # 얼굴 검출 시도 
        image, face,face_mo,imgh,imgy,imgx = face_detector(frame)
        try:            
            min_score = 999       #가장 낮은 점수로 예측된 사람의 점수
            min_score_name = ""   #가장 높은 점수로 예측된 사람의 이름
            
            #검출된 사진을 흑백으로 변환 
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            #위에서 학습한 모델로 예측시도
            for key, model in models.items():
                result = model.predict(face)                
                if min_score > result[1]:
                    min_score = result[1]
                    min_score_name = key
                    
            #min_score 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다.         
            if min_score < 500:
                confidence = int(100*(1-(min_score)/300))
                # 유사도 화면에 표시 
                display_string = str(confidence)+'%' + min_score_name 
            cv2.putText(image,display_string,(30,30), cv2.FONT_HERSHEY_COMPLEX,1,(51,97,231),2)
            
            if confidence > 85:      
            #85보다 크면 동일인물로 간주         
                cv2.imshow('Face Cropper', image)
                
            else:
            #85 이하면 타인으로 간주
                image[imgy:(imgy+imgh),imgx:(imgx+imgh)]=face_mo
                cv2.imshow('Face Cropper', image)
        except:
            #얼굴 검출 안됨 
            cv2.putText(image, "Face Not Found", (200, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass
        if cv2.waitKey(1)==13:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 학습 시작
    models = trains()
    # 실행
    run(models)